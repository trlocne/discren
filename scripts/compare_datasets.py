"""Collect the paired full-vs-no-LLM result for every dataset and compare them.

Reads the evaluation logs written by ``run_dataset.sh <dataset> eval`` and emits
both a human-readable summary and the LaTeX rows for the paper's main and
ablation tables, so the numbers in the manuscript come from the logs rather than
from transcription.

The comparison the paper rests on is not "is the gain positive" but "does the
gain behave the way the mechanism predicts". Clothing is visually distinctive
with attribute-dense titles; Sports is functional and more visually uniform. If
the LLM channel supplies genuine content information, its effect should be
*smaller* on Sports. If instead it is acting as extra capacity, there is no
reason for the two to differ in that direction. This script therefore reports the
cold-slice and aggregate effects side by side rather than one dataset at a time.

Usage::

    python compare_datasets.py
    python compare_datasets.py --datasets Clothing Sports --latex
"""

from __future__ import annotations

import argparse
import json
import os
import re

LOG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

RE_JSON = re.compile(r"^\[metrics-json\] (.*)$", re.M)

RE_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
    re.M,
)
ARMS = ("full", "wo_llm")

def parse_eval_log(path: str) -> dict[str, float] | None:
    """Return a metric dict from one eval log, or None if it has no results."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    payloads = RE_JSON.findall(text)
    if payloads:
        try:
            return json.loads(payloads[-1])["metrics"]
        except (json.JSONDecodeError, KeyError):
            pass

    metrics: dict[str, float] = {}
    for k, recall, prec, ndcg, mrr, cov, cold in RE_ROW.findall(text):
        metrics[f"recall@{k}"] = float(recall)
        metrics[f"precision@{k}"] = float(prec)
        metrics[f"ndcg@{k}"] = float(ndcg)
        metrics[f"mrr@{k}"] = float(mrr)
        metrics[f"coverage@{k}"] = float(cov)
        metrics[f"cold_recall@{k}"] = float(cold)
    return metrics or None

def collect(datasets: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for ds in datasets:
        arms: dict[str, dict[str, float]] = {}
        for arm in ARMS:
            path = os.path.join(LOG_ROOT, ds, f"eval_{arm}.log")
            metrics = parse_eval_log(path)
            if metrics is None:
                print(f"[skip] no results in logs/{ds}/eval_{arm}.log")
                continue
            arms[arm] = metrics
        if arms:
            out[ds] = arms
    return out

def rel(new: float, old: float) -> float:
    return 100.0 * (new - old) / old if old else float("nan")

def print_summary(runs, metrics_wanted):
    print("=" * 78)
    print("PER-DATASET RESULTS (test set)")
    print("=" * 78)
    for ds, arms in runs.items():
        print(f"\n  {ds}")
        header = f"    {'metric':<18}" + "".join(f"{a:>12}" for a in ARMS) + f"{'delta':>10}"
        print(header)
        for m in metrics_wanted:
            if not all(m in arms.get(a, {}) for a in ARMS):
                continue
            f, w = arms["full"][m], arms["wo_llm"][m]
            print(f"    {m:<18}{w:>12.4f}{f:>12.4f}{rel(f, w):>9.1f}%")

    both = [ds for ds, a in runs.items() if all(x in a for x in ARMS)]
    if len(both) < 2:
        print("\n(only one dataset has both arms; cross-dataset comparison skipped)")
        return

    print()
    print("=" * 78)
    print("CROSS-DATASET: does the LLM effect behave as the mechanism predicts?")
    print("=" * 78)
    print(f"\n  {'dataset':<12}{'R@20 delta':>14}{'coldR@20 delta':>18}{'ratio':>10}")
    for ds in both:
        a = runs[ds]
        agg = rel(a["full"]["recall@20"], a["wo_llm"]["recall@20"])
        cold_key = "cold_recall@20"
        if cold_key not in a["full"]:
            print(f"  {ds:<12}{agg:>13.1f}%{'n/a':>18}{'n/a':>10}")
            continue
        cold = rel(a["full"][cold_key], a["wo_llm"][cold_key])
        ratio = cold / agg if agg else float("nan")
        print(f"  {ds:<12}{agg:>13.1f}%{cold:>17.1f}%{ratio:>9.1f}x")

    print()
    print("  Reading: a ratio well above 1 means the gain concentrates in the cold")
    print("  slice, which is what routing content by reliability predicts. A ratio")
    print("  near 1 would mean the channel acts uniformly, i.e. as added capacity.")
    print("  Across datasets, a genuine content signal should be SMALLER on the")
    print("  less visually distinctive catalogue (Sports), not larger.")

def print_latex(runs):
    print()
    print("=" * 78)
    print("LATEX ROWS (paste into tab:main / tab:ablation)")
    print("=" * 78)
    print("\n% --- tab:main: one row per arm, R@20 / P@20 / N@20 per dataset ---")
    datasets = list(runs)
    for arm, label in (("wo_llm", r"\model (w/o LLM)"), ("full", r"\textbf{\model (Ours)}")):
        cells = []
        for ds in datasets:
            m = runs[ds].get(arm)
            if not m:
                cells += ["--", "--", "--"]
                continue
            cells += [f"{m.get('recall@20', 0):.4f}",
                      f"{m.get('precision@20', 0):.4f}",
                      f"{m.get('ndcg@20', 0):.4f}"]
        print(f"{label:<26} & " + " & ".join(cells) + r" \\")

    print("\n% --- tab:ablation: paired LLM ablation per dataset ---")
    for ds in datasets:
        a = runs[ds]
        if not all(x in a for x in ARMS):
            continue
        f, w = a["full"], a["wo_llm"]
        print(f"% {ds}")
        print(f"\\textbf{{Full model}} & {f.get('recall@20', 0):.4f} & "
              f"{f.get('ndcg@20', 0):.4f} & {f.get('cold_recall@20', 0):.4f} & "
              f"{f.get('coverage@20', 0):.4f} \\\\")
        print(f"w/o LLM stack & {w.get('recall@20', 0):.4f} & "
              f"{w.get('ndcg@20', 0):.4f} & {w.get('cold_recall@20', 0):.4f} & "
              f"{w.get('coverage@20', 0):.4f} \\\\")
        print(f"Relative change & {rel(f.get('recall@20', 0), w.get('recall@20', 1)):+.1f}\\% & "
              f"{rel(f.get('ndcg@20', 0), w.get('ndcg@20', 1)):+.1f}\\% & "
              f"{rel(f.get('cold_recall@20', 0), w.get('cold_recall@20', 1)):+.1f}\\% & "
              f"{rel(f.get('coverage@20', 0), w.get('coverage@20', 1)):+.1f}\\% \\\\")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=["Clothing", "Sports"])
    ap.add_argument("--metrics", nargs="*", default=[
        "recall@10", "recall@20", "ndcg@10", "ndcg@20",
        "precision@20", "mrr@20", "coverage@20", "cold_recall@20",
    ])
    ap.add_argument("--latex", action="store_true",
                    help="Also emit LaTeX table rows.")
    args = ap.parse_args()

    runs = collect(args.datasets)
    if not runs:
        raise SystemExit(
            "no evaluation logs found. Run:\n"
            "  bash run_dataset.sh Clothing eval\n"
            "  bash run_dataset.sh Sports eval"
        )

    print_summary(runs, args.metrics)
    if args.latex:
        print_latex(runs)

if __name__ == "__main__":
    main()
