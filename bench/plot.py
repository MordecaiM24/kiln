#!/usr/bin/env python3
"""Regenerate benchmark plots from committed JSON results."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BENCH_DIR = Path(__file__).resolve().parent


def load_all_records(results_dir):
    """Load records from every JSON file, keeping the newest run per key."""
    by_key = {}
    for path in sorted(results_dir.glob("*.json")):
        with path.open() as f:
            payload = json.load(f)
        ts = payload.get("env", {}).get("timestamp", "")
        for rec in payload.get("records", []):
            case_key = tuple(sorted(rec.get("case", {}).items()))
            key = (rec["suite"], rec["provider"], case_key)
            prev = by_key.get(key)
            if prev is None or ts >= prev["timestamp"]:
                by_key[key] = {"timestamp": ts, "record": rec}
    return [v["record"] for v in by_key.values()]


def _percentiles(raw_ms):
    arr = np.asarray(raw_ms, dtype=np.float64)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return float(q1), float(med), float(q3)


def _provider_label(provider, status):
    if status == "ok":
        return provider
    return f"{provider}: {status}"


def _status_style(status):
    if status == "ok":
        return {}
    return {"color": "0.6", "linestyle": "--", "marker": "x", "alpha": 0.7}


def plot_rmsnorm(records, outdir):
    groups = defaultdict(list)
    for rec in records:
        if rec["suite"] != "rmsnorm":
            continue
        c = rec["case"]
        groups[(c["dtype"], c["N"], c["mode"])].append(rec)

    for (dtype, n, mode), group in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        providers = sorted({r["provider"] for r in group})
        for provider in providers:
            rows = [r for r in group if r["provider"] == provider]
            status = rows[0]["status"]
            label = _provider_label(provider, status)
            style = _status_style(status)
            ok_rows = [r for r in rows if r["status"] == "ok" and r.get("raw_ms")]
            if not ok_rows:
                ax.plot([], [], label=label, **style)
                continue
            ms = sorted(ok_rows, key=lambda r: r["case"]["M"])
            xs = [r["case"]["M"] for r in ms]
            meds = []
            q1s = []
            q3s = []
            for r in ms:
                q1, med, q3 = _percentiles(r["raw_ms"])
                meds.append(med)
                q1s.append(q1)
                q3s.append(q3)
            (line,) = ax.plot(xs, meds, marker="o", label=label, **style)
            color = line.get_color()
            ax.fill_between(xs, q1s, q3s, alpha=0.2, color=color, linewidth=0)

        ax.set_xscale("log")
        ax.set_xlabel("M (rows)")
        ax.set_ylabel("median latency (ms)")
        ax.set_title(f"rmsnorm  dtype={dtype}  N={n}  mode={mode}")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fname = outdir / f"rmsnorm_{dtype}_N{n}_{mode}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)


def plot_sampling(records, outdir):
    groups = defaultdict(list)
    for rec in records:
        if rec["suite"] != "sampling":
            continue
        c = rec["case"]
        groups[(c["V"], c["p"])].append(rec)

    for (v, p), group in sorted(groups.items()):
        k_vals = sorted({r["case"]["k"] for r in group})
        nrows = len(k_vals)
        fig, axes = plt.subplots(nrows, 1, figsize=(9, 3 * nrows), sharex=True)
        if nrows == 1:
            axes = [axes]
        providers = sorted({r["provider"] for r in group})

        for ax, k in zip(axes, k_vals):
            sub = [r for r in group if r["case"]["k"] == k]
            b_vals = sorted({r["case"]["B"] for r in sub})
            x = np.arange(len(b_vals))
            width = 0.8 / max(len(providers), 1)

            for i, provider in enumerate(providers):
                rows = [r for r in sub if r["provider"] == provider]
                status = rows[0]["status"] if rows else "missing"
                label = _provider_label(provider, status)
                meds = []
                err_lo = []
                err_hi = []
                for b in b_vals:
                    match = [r for r in rows if r["case"]["B"] == b]
                    if not match or match[0]["status"] != "ok" or not match[0].get("raw_ms"):
                        meds.append(np.nan)
                        err_lo.append(0)
                        err_hi.append(0)
                        continue
                    q1, med, q3 = _percentiles(match[0]["raw_ms"])
                    meds.append(med)
                    err_lo.append(med - q1)
                    err_hi.append(q3 - med)

                offset = (i - (len(providers) - 1) / 2) * width
                style = _status_style(status)
                hatch = "///" if status != "ok" else None
                bars = ax.bar(
                    x + offset,
                    meds,
                    width,
                    label=label,
                    yerr=[err_lo, err_hi],
                    capsize=2,
                    hatch=hatch,
                    **{k: v for k, v in style.items() if k != "linestyle"},
                )
                if status != "ok":
                    for bar in bars:
                        bar.set_edgecolor("0.4")

            ax.set_xticks(x)
            ax.set_xticklabels([str(b) for b in b_vals])
            ax.set_ylabel("median (ms)")
            ax.set_title(f"k={k}")
            ax.grid(True, axis="y", alpha=0.3)

        axes[-1].set_xlabel("B (batch)")
        fig.suptitle(f"sampling  V={v}  p={p}", y=1.01)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=8)
        fig.tight_layout()
        fname = outdir / f"sampling_V{v}_p{p}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark JSON results.")
    parser.add_argument(
        "--results", type=Path, default=BENCH_DIR / "results", help="JSON input dir"
    )
    parser.add_argument(
        "--outdir", type=Path, default=BENCH_DIR / "plots", help="PNG output dir"
    )
    args = parser.parse_args()

    if not args.results.is_dir():
        raise SystemExit(f"results dir not found: {args.results}")

    records = load_all_records(args.results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_rmsnorm(records, args.outdir)
    plot_sampling(records, args.outdir)
    print(f"wrote plots to {args.outdir}")


if __name__ == "__main__":
    main()
