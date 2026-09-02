#!/usr/bin/env python3
"""Compare SpecForge vs vLLM DFlash draft logits.

Loads the two probe files produced by probe_vllm_draft_logits.py and
probe_specforge_draft_logits.py and prints a structured parity report.

Can be run under ANY Python environment that has torch and numpy installed.

Typical usage:

    python compare_draft_logits.py \\
        --specforge-probe /tmp/specforge_draft_probe.pt \\
        --vllm-probe      /tmp/vllm_draft_probe.pt \\
        --out             /tmp/parity_report.json \\
        [--topk 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D float vectors."""
    return F.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item()


def _topk_tokens(logits: torch.Tensor, k: int) -> list[int]:
    return logits.topk(k).indices.tolist()


def _scalar_stats(t: torch.Tensor) -> dict:
    t = t.float()
    return {
        "norm":     float(t.norm().item()),
        "abs_mean": float(t.abs().mean().item()),
        "max_abs":  float(t.abs().max().item()),
        "std":      float(t.std().item()) if t.numel() > 1 else 0.0,
    }


def _compare_tensors(a: torch.Tensor, b: torch.Tensor,
                     label: str) -> dict:
    """Per-token or per-row cosine / max-abs comparison."""
    assert a.shape == b.shape, f"{label}: shape mismatch {a.shape} vs {b.shape}"
    a, b = a.float(), b.float()
    if a.ndim == 1:
        # Single vector
        return {
            "cosine":  _cos(a, b),
            "max_abs": float((a - b).abs().max().item()),
            "mean_abs":float((a - b).abs().mean().item()),
        }
    # 2-D: [rows, dim]
    rows = a.shape[0]
    per_row_cos   = [_cos(a[i], b[i]) for i in range(rows)]
    per_row_maxabs = [(a[i] - b[i]).abs().max().item() for i in range(rows)]
    diff = a - b
    return {
        "shape":         list(a.shape),
        "mean_cosine":   float(sum(per_row_cos) / len(per_row_cos)),
        "min_cosine":    float(min(per_row_cos)),
        "per_row_cos":   [round(c, 6) for c in per_row_cos],
        "mean_max_abs":  float(sum(per_row_maxabs) / len(per_row_maxabs)),
        "global_max_abs":float(diff.abs().max().item()),
        "global_mean_abs":float(diff.abs().mean().item()),
    }


def _draft_logit_report(
    sf_logits: torch.Tensor,    # [num_spec, vocab]
    vllm_logits: torch.Tensor,  # [num_spec, vocab]
    topk: int = 10,
) -> dict:
    """Full comparison report for the draft logit tensor."""
    assert sf_logits.shape == vllm_logits.shape, (
        f"Draft logit shape mismatch: sf={sf_logits.shape} vllm={vllm_logits.shape}"
    )
    num_spec, vocab = sf_logits.shape
    sf, vl = sf_logits.float(), vllm_logits.float()

    per_depth = []
    for d in range(num_spec):
        sf_d, vl_d = sf[d], vl[d]
        sf_top1  = int(sf_d.argmax())
        vl_top1  = int(vl_d.argmax())
        sf_topk  = _topk_tokens(sf_d, topk)
        vl_topk  = _topk_tokens(vl_d, topk)
        overlap  = len(set(sf_topk) & set(vl_topk))
        # KL divergence: KL(sf || vllm)  (both softmax-normalized)
        sf_lp = F.log_softmax(sf_d, dim=-1)
        vl_lp = F.log_softmax(vl_d, dim=-1)
        kl = F.kl_div(vl_lp, sf_lp.exp(), reduction="sum").item()
        per_depth.append({
            "depth":        d,
            "cosine":       round(_cos(sf_d, vl_d), 6),
            "max_abs_diff": round(float((sf_d - vl_d).abs().max().item()), 6),
            "mean_abs_diff":round(float((sf_d - vl_d).abs().mean().item()), 6),
            "kl_sf_vs_vllm":round(kl, 6),
            "sf_top1":      sf_top1,
            "vllm_top1":    vl_top1,
            "top1_match":   sf_top1 == vl_top1,
            f"top{topk}_overlap": overlap,
            f"top{topk}_overlap_frac": round(overlap / topk, 3),
        })

    top1_match_rate = sum(r["top1_match"] for r in per_depth) / num_spec
    mean_cos  = sum(r["cosine"] for r in per_depth) / num_spec
    mean_kl   = sum(r["kl_sf_vs_vllm"] for r in per_depth) / num_spec
    mean_topk = sum(r[f"top{topk}_overlap_frac"] for r in per_depth) / num_spec

    return {
        "num_draft_depths": num_spec,
        "vocab_size":        vocab,
        "summary": {
            "top1_match_rate":          round(top1_match_rate, 4),
            "mean_cosine":              round(mean_cos, 6),
            "mean_kl_sf_vs_vllm":       round(mean_kl, 6),
            f"mean_top{topk}_overlap_frac": round(mean_topk, 4),
            "verdict": (
                "ALIGNED" if mean_cos > 0.9999 and mean_kl < 1e-4
                else "CLOSE"  if mean_cos > 0.999  and mean_kl < 0.01
                else "DRIFT"  if mean_cos > 0.99
                else "MISALIGNED"
            ),
        },
        "per_depth": per_depth,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Compare SpecForge vs vLLM draft logits")
    p.add_argument(
        "--specforge-probe", required=True,
        help="specforge_draft_probe.pt from probe_specforge_draft_logits.py"
    )
    p.add_argument(
        "--vllm-probe", required=True,
        help="vllm_draft_probe.pt from probe_vllm_draft_logits.py (for raw_target_hidden)"
    )
    p.add_argument("--out", default=None, help="Optional path to save parity_report.json")
    p.add_argument("--topk", type=int, default=10, help="K for top-k overlap metric")
    return p.parse_args()


def main():
    args = _parse_args()

    print(f"[compare] Loading specforge probe: {args.specforge_probe}")
    sf_probe = torch.load(args.specforge_probe, weights_only=True)
    print(f"[compare] Loading vLLM probe:       {args.vllm_probe}")
    vllm_probe = torch.load(args.vllm_probe, weights_only=True)

    report: dict = {"sections": {}}

    # ── 1. Target hidden state parity ────────────────────────────────────────
    # Both probes contain raw_target_hidden_states (identical tensor since we
    # loaded it from the vLLM probe in the SpecForge script), but let's verify.
    raw_sf   = sf_probe.get("raw_target_hidden_states")
    raw_vllm = vllm_probe.get("raw_target_hidden_states")
    if raw_sf is not None and raw_vllm is not None:
        tol_ok = torch.allclose(raw_sf.float(), raw_vllm.float(), atol=1e-5)
        report["sections"]["raw_target_hidden_passthrough"] = {
            "allclose_1e5": bool(tol_ok),
            "max_diff": float((raw_sf.float() - raw_vllm.float()).abs().max().item()),
            "note": (
                "Should be identical (sf loaded it from vllm probe). "
                "Non-zero diff suggests a file mismatch."
            ),
        }
        if not tol_ok:
            print("[compare] WARNING: raw_target_hidden_states differ between files!")
    else:
        report["sections"]["raw_target_hidden_passthrough"] = {"skipped": True}

    # ── 2. FC-output parity  (SpecForge fc_output  ↔  vLLM combined_target_hs)
    # SpecForge fc_output = F.linear(raw, fc.weight)              [num_ctx, H]
    # vLLM combined      = drafter.model.combine_hidden_states()  [num_ctx, H]
    # Both should be identical if weights are the same.
    sf_fc   = sf_probe.get("fc_output")
    _vl_fc_candidate = sf_probe.get("vllm_combined_target_hidden_states")
    vl_fc = (
        _vl_fc_candidate
        if _vl_fc_candidate is not None
        else vllm_probe.get("combined_target_hidden_states")
    )
    if sf_fc is not None and vl_fc is not None and sf_fc.shape == vl_fc.shape:
        fc_cmp = _compare_tensors(sf_fc, vl_fc, "fc_output")
        report["sections"]["fc_output_parity"] = {
            **fc_cmp,
            "verdict": (
                "ALIGNED"     if fc_cmp["mean_cosine"] > 0.9999
                else "CLOSE"  if fc_cmp["mean_cosine"] > 0.999
                else "MISALIGNED"
            ),
        }
        print(f"[compare] FC output  mean_cos={fc_cmp.get('mean_cosine', fc_cmp.get('cosine')):.6f}"
              f"  max_abs={fc_cmp['global_max_abs']:.6f}")
    else:
        reason = []
        if sf_fc is None:  reason.append("sf_fc missing")
        if vl_fc is None:  reason.append("vl_fc missing")
        if sf_fc is not None and vl_fc is not None and sf_fc.shape != vl_fc.shape:
            reason.append(f"shape mismatch {sf_fc.shape} vs {vl_fc.shape}")
        report["sections"]["fc_output_parity"] = {"skipped": True, "reason": reason}
        print(f"[compare] FC output parity skipped: {reason}")

    # ── 3. Draft logit parity ────────────────────────────────────────────────
    sf_logits   = sf_probe.get("draft_logits")           # [num_spec, vocab]
    _vllm_logits_candidate = sf_probe.get("vllm_draft_logits_req0")
    vllm_logits = (
        _vllm_logits_candidate
        if _vllm_logits_candidate is not None
        else vllm_probe.get("draft_logits_req0")
    )

    if sf_logits is None or vllm_logits is None:
        print("[compare] ERROR: draft logits missing from one or both probe files")
        report["sections"]["draft_logit_parity"] = {
            "error": "missing_tensors",
            "sf_present":   sf_logits is not None,
            "vllm_present": vllm_logits is not None,
        }
    elif sf_logits.shape != vllm_logits.shape:
        print(
            f"[compare] ERROR: shape mismatch sf={list(sf_logits.shape)} "
            f"vllm={list(vllm_logits.shape)}"
        )
        report["sections"]["draft_logit_parity"] = {
            "error": "shape_mismatch",
            "sf_shape":   list(sf_logits.shape),
            "vllm_shape": list(vllm_logits.shape),
        }
    else:
        logit_report = _draft_logit_report(sf_logits, vllm_logits, topk=args.topk)
        report["sections"]["draft_logit_parity"] = logit_report
        s = logit_report["summary"]
        print("\n" + "=" * 60)
        print("DRAFT LOGIT PARITY REPORT")
        print("=" * 60)
        print(f"  Verdict           : {s['verdict']}")
        print(f"  Mean cosine       : {s['mean_cosine']:.6f}")
        print(f"  Top-1 match rate  : {s['top1_match_rate']:.4f}  "
              f"({int(s['top1_match_rate'] * logit_report['num_draft_depths'])}/"
              f"{logit_report['num_draft_depths']} depths)")
        print(f"  Mean KL(sf||vllm) : {s['mean_kl_sf_vs_vllm']:.6f}")
        print(f"  Mean top-{args.topk} overlap: {s[f'mean_top{args.topk}_overlap_frac']:.4f}")
        print()
        print(f"  {'depth':>5}  {'cosine':>10}  {'max_abs':>10}  "
              f"{'KL':>10}  {'sf_top1':>8}  {'vl_top1':>8}  match")
        print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  "
              f"{'-'*8}  {'-'*8}  -----")
        for row in logit_report["per_depth"]:
            print(
                f"  {row['depth']:>5}  {row['cosine']:>10.6f}  "
                f"{row['max_abs_diff']:>10.6f}  "
                f"{row['kl_sf_vs_vllm']:>10.6f}  "
                f"{row['sf_top1']:>8}  {row['vllm_top1']:>8}  "
                f"{'✓' if row['top1_match'] else '✗'}"
            )
        print("=" * 60)

    # ── 4. Overall verdict ───────────────────────────────────────────────────
    dl_section = report["sections"].get("draft_logit_parity", {})
    dl_verdict = dl_section.get("summary", {}).get("verdict", "UNKNOWN")
    fc_verdict = report["sections"].get("fc_output_parity", {}).get("verdict", "UNKNOWN")
    overall = (
        "ALIGNED" if dl_verdict == "ALIGNED" and fc_verdict in ("ALIGNED", "UNKNOWN")
        else "CLOSE"  if dl_verdict == "CLOSE"
        else "MISALIGNED"
    )
    report["overall_verdict"] = overall
    report["fc_verdict"] = fc_verdict
    report["draft_logit_verdict"] = dl_verdict

    print(f"\n[compare] Overall verdict: {overall}")
    print(f"[compare] FC output:       {fc_verdict}")
    print(f"[compare] Draft logits:    {dl_verdict}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Serialise any tensors to lists for JSON
        def _serialise(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _serialise(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialise(v) for v in obj]
            return obj

        with out_path.open("w") as f:
            json.dump(_serialise(report), f, indent=2)
        print(f"[compare] Report saved to {out_path}")

    # Exit with non-zero code if misaligned (useful for CI)
    if overall == "MISALIGNED":
        sys.exit(1)


if __name__ == "__main__":
    main()
