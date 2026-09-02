#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import csv
from pathlib import Path


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.replace(",", " ").split()]


def read_metrics(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not path.is_file():
        return metrics
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metrics[key.strip()] = value.strip()
    return metrics


def as_float(metrics: dict[str, str], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def metric_report_path(
    profiler_dir: Path,
    *,
    label: str,
    mode: str,
    tp_size: int,
    batch_size: int,
) -> Path:
    return profiler_dir / label / mode / f"tp{tp_size}" / f"bs{batch_size}" / "metrics_report.txt"


def format_latex_cell(tps: float | None, speedup: float | None) -> str:
    if tps is None or speedup is None:
        return "--"
    return f"{tps:.1f} (${speedup:.2f} \\times$)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Math-500 DFlash batch-size/tree-budget sweep metrics."
    )
    parser.add_argument("--profiler-dir", required=True, type=Path)
    parser.add_argument("--batch-sizes", default="1 2 4 8 16")
    parser.add_argument("--tree-budgets", default="16 32 64 128 256")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tree-kv-layout", default="physical")
    parser.add_argument(
        "--metric",
        default="e2e_throughput_tok_s",
        choices=["e2e_throughput_tok_s", "benchmark_tok_s"],
        help="Metric used for TPS and speedup.",
    )
    args = parser.parse_args()

    batch_sizes = parse_int_list(args.batch_sizes)
    tree_budgets = parse_int_list(args.tree_budgets)

    rows: list[dict[str, str]] = []
    latex_rows: list[str] = []
    for batch_size in batch_sizes:
        ar_path = metric_report_path(
            args.profiler_dir,
            label=f"ar_bsz{batch_size}",
            mode="ar",
            tp_size=args.tp_size,
            batch_size=batch_size,
        )
        ar_metrics = read_metrics(ar_path)
        ar_tps = as_float(ar_metrics, args.metric)

        latex_cells = [str(batch_size)]
        for budget in tree_budgets:
            dflash_label = f"budget{budget}_bsz{batch_size}_{args.tree_kv_layout}"
            dflash_path = metric_report_path(
                args.profiler_dir,
                label=dflash_label,
                mode="dflash",
                tp_size=args.tp_size,
                batch_size=batch_size,
            )
            dflash_metrics = read_metrics(dflash_path)
            dflash_tps = as_float(dflash_metrics, args.metric)
            speedup = dflash_tps / ar_tps if dflash_tps is not None and ar_tps else None
            rows.append(
                {
                    "batch_size": str(batch_size),
                    "tree_budget": str(budget),
                    "metric": args.metric,
                    "ar_tps": "" if ar_tps is None else f"{ar_tps:.6f}",
                    "dflash_tps": "" if dflash_tps is None else f"{dflash_tps:.6f}",
                    "speedup": "" if speedup is None else f"{speedup:.6f}",
                    "ar_num_samples": ar_metrics.get("num_samples", ""),
                    "dflash_num_samples": dflash_metrics.get("num_samples", ""),
                    "ar_report": str(ar_path),
                    "dflash_report": str(dflash_path),
                }
            )
            latex_cells.append(format_latex_cell(dflash_tps, speedup))
        latex_rows.append(" & ".join(latex_cells) + r" \\")

    args.profiler_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.profiler_dir / "run_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    budget_headers = " & ".join(f"Budget {budget}" for budget in tree_budgets)
    latex_lines = [
        r"\begin{tabular}{" + "c" * (len(tree_budgets) + 1) + "}",
        r"\toprule",
        f"Batch Size & {budget_headers} " + r"\\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    latex_path = args.profiler_dir / "run_summary_latex.tex"
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    print(f"Wrote CSV summary to: {csv_path}")
    print(f"Wrote LaTeX summary to: {latex_path}")


if __name__ == "__main__":
    main()
