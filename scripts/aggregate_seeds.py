"""Reduce a multi-seed evaluation sweep to mean +- std and a paired test.

Reads the ``[metrics-json]`` lines emitted by ``eval_modal.py`` from
``logs/seeds/eval_<run>_seed<N>.log`` and reports, per metric:

* mean and sample standard deviation across seeds,
* the full-vs-no-LLM difference,
* a **paired** t-test over seeds (same seed = same data order = same
  initialization, so the runs pair up naturally and the paired test is the
  correct one),
* the seed-to-seed spread of the baseline, which is the honest yardstick for
  "is this gap real".

The point of this script is to stop a +1.2% single-seed difference from being
reported as an effect when the baseline's own seed noise may be larger.

Usage::

    python aggregate_seeds.py
    python aggregate_seeds.py --metric recall@20 ndcg@20 cold_recall@20
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "seeds")
RE_JSON = re.compile(r"^\[metrics-json\] (.*)$", re.M)
RE_NAME = re.compile(r"eval_(?P<run>.+)_seed(?P<seed>-?\d+)\.log$")

DEFAULT_METRICS = ["recall@20", "ndcg@20", "cold_recall@20", "coverage@20"]


def collect(log_dir: str) -> dict[str, dict[int, dict[str, float]]]:
    """Return ``{run_name: {seed: {metric: value}}}`` parsed from the logs."""
    runs: dict[str, dict[int, dict[str, float]]] = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "eval_*_seed*.log"))):
        m = RE_NAME.search(os.path.basename(path))
        if not m:
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        payloads = RE_JSON.findall(text)
        if not payloads:
            print(f"[skip] no [metrics-json] line in {os.path.basename(path)}")
            continue
        try:
            data = json.loads(payloads[-1])
        except json.JSONDecodeError:
            print(f"[skip] malformed metrics JSON in {os.path.basename(path)}")
            continue
        runs.setdefault(m.group("run"), {})[int(m.group("seed"))] = data["metrics"]
    return runs


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and *sample* standard deviation (ddof=1); std is 0.0 for n < 2."""
    n = len(values)
    mu = sum(values) / n
    if n < 2:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return mu, math.sqrt(var)


def paired_t(diffs: list[float]) -> tuple[float, int]:
    """Paired t statistic and degrees of freedom for a list of differences."""
    n = len(diffs)
    if n < 2:
        return float("nan"), 0
    mu, sd = mean_std(diffs)
    if sd == 0.0:
        return float("inf") if mu != 0 else 0.0, n - 1
    return mu / (sd / math.sqrt(n)), n - 1


# Two-sided 95% critical values for small samples; enough for a 3-10 seed sweep
# and avoids a scipy dependency inside the Modal image.
_T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
              6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", default=LOG_DIR)
    ap.add_argument("--metric", nargs="*", default=DEFAULT_METRICS)
    ap.add_argument("--treatment", default="full")
    ap.add_argument("--baseline", default="wo_llm")
    args = ap.parse_args()

    runs = collect(args.log_dir)
    if not runs:
        raise SystemExit(
            f"no seeded eval logs found in {args.log_dir}\n"
            "run:  bash run_seeds.sh && bash run_seeds.sh eval"
        )

    print("=" * 78)
    print("PER-RUN SUMMARY (mean +- std over seeds)")
    print("=" * 78)
    for run, per_seed in sorted(runs.items()):
        seeds = sorted(per_seed)
        print(f"\n  {run}   seeds={seeds}  (n={len(seeds)})")
        if len(seeds) < 3:
            print("    WARNING: fewer than 3 seeds — the std below is not meaningful.")
        for metric in args.metric:
            vals = [per_seed[s][metric] for s in seeds if metric in per_seed[s]]
            if not vals:
                continue
            mu, sd = mean_std(vals)
            print(f"    {metric:<18} {mu:.4f} +- {sd:.4f}   "
                  f"[{min(vals):.4f}, {max(vals):.4f}]")

    treat = runs.get(args.treatment)
    base = runs.get(args.baseline)
    if not treat or not base:
        print(f"\n(no paired comparison: need both '{args.treatment}' and "
              f"'{args.baseline}')")
        return

    shared = sorted(set(treat) & set(base))
    if not shared:
        print("\n(no paired comparison: the two runs share no seed)")
        return

    print()
    print("=" * 78)
    print(f"PAIRED COMPARISON  {args.treatment} - {args.baseline}   "
          f"seeds={shared} (n={len(shared)})")
    print("=" * 78)
    for metric in args.metric:
        diffs = [treat[s][metric] - base[s][metric]
                 for s in shared
                 if metric in treat[s] and metric in base[s]]
        if not diffs:
            continue
        mu_d, sd_d = mean_std(diffs)
        base_vals = [base[s][metric] for s in shared if metric in base[s]]
        _, base_sd = mean_std(base_vals)
        base_mu = sum(base_vals) / len(base_vals)
        rel = 100.0 * mu_d / base_mu if base_mu else float("nan")

        t, df = paired_t(diffs)
        crit = _T_CRIT_95.get(df)
        if df == 0:
            verdict = "n/a (need >= 2 seeds)"
        elif not math.isfinite(t):
            # sd == 0: every seed produced the exact same delta. Real training
            # runs never do this, so treat it as a fixture/parsing artifact
            # rather than as infinitely strong evidence.
            verdict = "degenerate (zero variance across seeds — check the logs)"
        elif crit is None:
            verdict = f"n/a (no critical value for df={df})"
        elif abs(t) > crit:
            verdict = "SIGNIFICANT at 95%"
        else:
            verdict = "NOT significant at 95%"
        t_str = "inf" if not math.isfinite(t) else f"{t:.3f}"

        # The decisive sanity check: an effect smaller than the baseline's own
        # seed-to-seed spread is not distinguishable from noise.
        noise_flag = ""
        if base_sd > 0 and abs(mu_d) < base_sd:
            noise_flag = "  <-- effect is SMALLER than baseline seed noise"

        print(f"\n  {metric}")
        print(f"    delta        {mu_d:+.4f} +- {sd_d:.4f}  ({rel:+.2f}%)")
        print(f"    baseline     {base_mu:.4f} +- {base_sd:.4f} (seed noise)")
        print(f"    paired t     t={t_str}  df={df}  ->  {verdict}{noise_flag}")

    print()
    print("Report the mean +- std and the paired test in the paper. Do NOT report")
    print("a single-seed delta as an effect.")


if __name__ == "__main__":
    main()
