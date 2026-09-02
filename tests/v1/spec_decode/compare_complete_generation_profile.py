from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_summary(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_complete_generation(
    *,
    reference_summary_path: str,
    vllm_summary_path: str,
    output_json: str | None = None,
) -> dict[str, Any]:
    ref = _load_summary(reference_summary_path)
    cand = _load_summary(vllm_summary_path)

    ref_samples = {int(s["sample_index"]): s for s in ref.get("samples", [])}
    cand_samples = {int(s["sample_index"]): s for s in cand.get("samples", [])}
    sample_indices = sorted(set(ref_samples).intersection(cand_samples))

    per_sample = []
    for sample_index in sample_indices:
        rs = ref_samples[sample_index]
        cs = cand_samples[sample_index]
        ref_d0 = float(rs.get("d0_acceptance_rate", 0.0))
        cand_d0 = float(cs.get("d0_acceptance_rate", 0.0))
        per_sample.append(
            {
                "sample_index": sample_index,
                "reference_d0_acceptance_rate": ref_d0,
                "vllm_d0_acceptance_rate": cand_d0,
                "d0_acceptance_rate_diff": cand_d0 - ref_d0,
                "reference_num_output_tokens": rs.get("num_output_tokens"),
                "vllm_num_output_tokens": cs.get("total_output_tokens"),
                "reference_time_per_output_token": rs.get("time_per_output_token"),
                "vllm_time_per_output_token": cs.get("mean_time_per_output_token_s"),
                "reference_output_text": rs.get("output_text", ""),
                "vllm_output_text": cs.get("output_text", ""),
                "reference_acceptance_lengths": rs.get("acceptance_lengths", []),
                "vllm_per_pos_acceptance_rate": cs.get("per_pos_acceptance_rate", []),
                "reference_per_pos_acceptance_rate": rs.get(
                    "per_pos_acceptance_rate", []
                ),
            }
        )

    summary = {
        "reference_summary": reference_summary_path,
        "vllm_summary": vllm_summary_path,
        "reference_d0_acceptance_rate": ref.get("d0_acceptance_rate"),
        "vllm_d0_acceptance_rate": cand.get("d0_acceptance_rate"),
        "d0_acceptance_rate_diff": (
            float(cand.get("d0_acceptance_rate", 0.0))
            - float(ref.get("d0_acceptance_rate", 0.0))
        ),
        "samples": per_sample,
    }

    if output_json is not None:
        Path(output_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare reference and vLLM complete-generation profiling summaries.",
    )
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--vllm-summary", required=True)
    parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = compare_complete_generation(
        reference_summary_path=args.reference_summary,
        vllm_summary_path=args.vllm_summary,
        output_json=args.output_json,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
