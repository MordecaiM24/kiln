#!/usr/bin/env python3
"""Regenerate the figures in bench/plots/ from the raw JSON in bench/results/.

Nothing is hard-coded from a run: every point is the median of a record's
`raw_ms`, every band is its interquartile range, and when a case appears in
more than one results file the newest record wins (so re-running after a new
benchmark run updates the figures in place).

Outputs
    sampling_V32768.png, sampling_V131072.png   core sampling matrix, one panel per (p, k)
    sampling_stress.png                          the six stress cases
    rmsnorm_float16.png, rmsnorm_bfloat16.png    core RMSNorm matrix, one panel per (mode, N)
    rmsnorm_stress.png                           the eight stress cases
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

matplotlib.use("Agg")

BENCH_DIR = Path(__file__).resolve().parent

# One color per provider, fixed for the whole repo so a reader who learns
# "Kiln is blue" is never misled between figures. Slots 1-4 of the palette in
# docs/benchmarking.md's figure conventions; validated for adjacent-pair
# colorblind separation on the light surface.
PROVIDERS = {
    "kiln":                   ("Kiln",                       "#2a78d6"),
    "eager":                  ("PyTorch eager",              "#eb6834"),
    "eager_hf_chain":         ("PyTorch eager chain",        "#eb6834"),
    "torch_compile":          ("torch.compile",              "#1baf7a"),
    "torch_compile_hf_chain": ("torch.compile (same chain)", "#1baf7a"),
    "liger":                  ("Liger-Kernel",               "#eda100"),
    "vllm_sampler":           ("vLLM sampler",               "#eda100"),
}
PROVIDER_ORDER = ["kiln", "eager", "eager_hf_chain", "torch_compile",
                  "torch_compile_hf_chain", "liger", "vllm_sampler"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "text.color": INK,
})


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def load_all_records(results_dir):
    """Newest record per (suite, provider, case) across every JSON file."""
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


def _stats_us(rec):
    """(q1, median, q3) in microseconds, or None if the record has no timings."""
    if rec["status"] != "ok" or not rec.get("raw_ms"):
        return None
    arr = np.asarray(rec["raw_ms"], dtype=np.float64) * 1000.0
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return float(q1), float(med), float(q3)


def _providers_present(records):
    present = {r["provider"] for r in records}
    return [p for p in PROVIDER_ORDER if p in present]


# ----------------------------------------------------------------------------
# chrome
# ----------------------------------------------------------------------------

def _fmt_thousands(x, _pos):
    return f"{x:,.0f}" if x >= 1 else f"{x:g}"


def _style_axes(ax, *, xlabel=None, ylabel=None, xlog=True):
    ax.set_yscale("log")
    if xlog:
        ax.set_xscale("log", base=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", which="major")
    ax.grid(False, axis="x")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(length=3, width=0.8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _line(ax, xs, stats, color):
    q1 = [s[0] for s in stats]
    med = [s[1] for s in stats]
    q3 = [s[2] for s in stats]
    ax.fill_between(xs, q1, q3, color=color, alpha=0.15, linewidth=0)
    ax.plot(
        xs, med, color=color, linewidth=2, solid_capstyle="round",
        marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
        zorder=3,
    )


def _legend(fig, providers, unsupported=()):
    handles, labels = [], []
    for p in providers:
        name, color = PROVIDERS[p]
        if p in unsupported:
            handles.append(Line2D([], [], linestyle="none", marker="x", color=MUTED, markersize=6))
            labels.append(f"{name}: not installable on the benchmark host")
        else:
            handles.append(Line2D([], [], color=color, linewidth=2, marker="o", markersize=6,
                                  markeredgecolor=SURFACE, markeredgewidth=1.2))
            labels.append(name)
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.985), fontsize=9, handlelength=2.2, columnspacing=1.6)


def _footer(fig, text):
    fig.text(0.01, 0.005, text, fontsize=7.5, color=MUTED, ha="left", va="bottom")


# ----------------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------------

def plot_sampling_core(records, outdir, V):
    core = [r for r in records if r["suite"] == "sampling" and r["case"]["V"] == V
            and r["case"]["dtype"] == "float16" and r["case"]["temperature"] == 0.8
            and r["case"]["k"] in (1, 50, 500) and r["case"]["B"] in (1, 32, 256)]
    if not core:
        return
    providers = _providers_present(core)
    unsupported = {p for p in providers
                   if all(r["status"] != "ok" for r in core if r["provider"] == p)}
    p_vals = sorted({r["case"]["p"] for r in core})
    k_vals = sorted({r["case"]["k"] for r in core})

    fig, axes = plt.subplots(len(p_vals), len(k_vals),
                             figsize=(3.3 * len(k_vals), 2.9 * len(p_vals) + 0.9),
                             sharex=True, sharey=True, squeeze=False)
    for i, p in enumerate(p_vals):
        for j, k in enumerate(k_vals):
            ax = axes[i][j]
            sub = [r for r in core if r["case"]["p"] == p and r["case"]["k"] == k]
            for prov in providers:
                rows = sorted((r for r in sub if r["provider"] == prov), key=lambda r: r["case"]["B"])
                pts = [(r["case"]["B"], _stats_us(r)) for r in rows]
                pts = [(b, s) for b, s in pts if s is not None]
                if pts:
                    _line(ax, [b for b, _ in pts], [s for _, s in pts], PROVIDERS[prov][1])
            _style_axes(ax,
                        xlabel="batch size" if i == len(p_vals) - 1 else None,
                        ylabel="median latency (µs)" if j == 0 else None)
            ax.set_xticks([1, 32, 256])
            ax.set_xticklabels(["1", "32", "256"])
            ax.set_title(f"top-k = {k:,}   top-p = {p}", loc="left", color=INK)
            ax.tick_params(axis="x", which="minor", bottom=False)

    _legend(fig, providers, unsupported)
    fig.suptitle(f"Fused top-k / top-p sampling, vocabulary {V:,} (fp16, temperature 0.8)",
                 x=0.01, ha="left", fontsize=11, color=INK, y=1.03)
    _footer(fig, "Medians of ≥100 timed iterations (triton.testing.do_bench); band = interquartile range. "
                 "NVIDIA L40S. Log axes.")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(outdir / f"sampling_V{V}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _dot_plot(rows, providers, unsupported, title, xlabel, outpath, footer):
    """One row per case, one dot per provider, log x. `rows` = [(label, {provider: stats})]."""
    n = len(rows)
    fig, ax = plt.subplots(figsize=(8.4, 0.52 * n + 1.6))
    ys = np.arange(n)[::-1]
    for y, (label, per_prov) in zip(ys, rows):
        ax.axhline(y, color=GRID, linewidth=0.8, zorder=1)
        for prov in providers:
            s = per_prov.get(prov)
            if s is None:
                continue
            ax.plot([s[0], s[2]], [y, y], color=PROVIDERS[prov][1], linewidth=2, alpha=0.35, zorder=2)
            ax.plot(s[1], y, marker="o", markersize=7, color=PROVIDERS[prov][1],
                    markeredgecolor=SURFACE, markeredgewidth=1.2, linestyle="none", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _ in rows], color=INK_2)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    ax.xaxis.set_minor_formatter(NullFormatter())
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, width=0.8)
    ax.grid(True, axis="x", which="major")
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.7, n - 0.3)
    _legend(fig, providers, unsupported)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK, y=1.02)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_sampling_stress(records, outdir):
    core_keys = {(1, 32768), (1, 131072), (32, 32768), (32, 131072), (256, 32768), (256, 131072)}

    def is_core(c):
        return (c["dtype"] == "float16" and c["temperature"] == 0.8 and c["k"] in (1, 50, 500)
                and (c["B"], c["V"]) in core_keys and c["p"] in (0.9, 1.0))

    stress = [r for r in records if r["suite"] == "sampling" and not is_core(r["case"])]
    if not stress:
        return
    providers = _providers_present(stress)
    unsupported = {p for p in providers
                   if all(r["status"] != "ok" for r in stress if r["provider"] == p)}
    by_case = defaultdict(dict)
    for r in stress:
        c = r["case"]
        label = (f"{c['dtype']}  B={c['B']}  V={c['V']:,}  k={c['k']:,}  p={c['p']}"
                 + (f"  T={c['temperature']}" if c["temperature"] != 0.8 else ""))
        s = _stats_us(r)
        if s is not None:
            by_case[label][r["provider"]] = s
    rows = sorted(by_case.items(), key=lambda kv: -max(s[1] for s in kv[1].values()))
    _dot_plot(rows, providers, unsupported,
              "Fused top-k / top-p sampling, stress cases", "median latency (µs), log scale",
              outdir / "sampling_stress.png",
              "Dot = median of ≥100 iterations; bar = interquartile range. NVIDIA L40S. T = temperature (0.8 unless shown).")


# ----------------------------------------------------------------------------
# rmsnorm
# ----------------------------------------------------------------------------

CORE_M = (1, 16, 256, 4096, 16384)
CORE_N = (1024, 4096, 8192)


def plot_rmsnorm_core(records, outdir, dtype):
    core = [r for r in records if r["suite"] == "rmsnorm" and r["case"]["dtype"] == dtype
            and r["case"]["M"] in CORE_M and r["case"]["N"] in CORE_N]
    if not core:
        return
    providers = _providers_present(core)
    modes = [m for m in ("fwd", "fwd_bwd") if any(r["case"]["mode"] == m for r in core)]
    mode_name = {"fwd": "forward", "fwd_bwd": "forward + backward"}

    fig, axes = plt.subplots(len(modes), len(CORE_N),
                             figsize=(3.3 * len(CORE_N), 2.9 * len(modes) + 0.9),
                             sharex=True, sharey=True, squeeze=False)
    for i, mode in enumerate(modes):
        for j, n in enumerate(CORE_N):
            ax = axes[i][j]
            sub = [r for r in core if r["case"]["mode"] == mode and r["case"]["N"] == n]
            for prov in providers:
                rows = sorted((r for r in sub if r["provider"] == prov), key=lambda r: r["case"]["M"])
                pts = [(r["case"]["M"], _stats_us(r)) for r in rows]
                pts = [(m, s) for m, s in pts if s is not None]
                if pts:
                    _line(ax, [m for m, _ in pts], [s for _, s in pts], PROVIDERS[prov][1])
            _style_axes(ax,
                        xlabel="rows (M)" if i == len(modes) - 1 else None,
                        ylabel="median latency (µs)" if j == 0 else None)
            ax.set_xticks(list(CORE_M))
            ax.set_xticklabels(["1", "16", "256", "4k", "16k"])
            ax.tick_params(axis="x", which="minor", bottom=False)
            ax.set_title(f"hidden size {n:,}   {mode_name[mode]}", loc="left", color=INK)

    _legend(fig, providers)
    fig.suptitle(f"RMSNorm, {dtype}", x=0.01, ha="left", fontsize=11, color=INK, y=1.03)
    _footer(fig, "Medians of ≥100 timed iterations (triton.testing.do_bench); band = interquartile range. "
                 "NVIDIA L40S. Log axes. The Liger forward point at M=4,096, N=4,096 is a known "
                 "provider-order artifact (docs/benchmarking.md).")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(outdir / f"rmsnorm_{dtype}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_rmsnorm_stress(records, outdir):
    stress = [r for r in records if r["suite"] == "rmsnorm"
              and not (r["case"]["M"] in CORE_M and r["case"]["N"] in CORE_N
                       and r["case"]["dtype"] in ("float16", "bfloat16"))]
    if not stress:
        return
    providers = _providers_present(stress)
    mode_name = {"fwd": "fwd", "fwd_bwd": "fwd+bwd"}
    by_case = defaultdict(dict)
    for r in stress:
        c = r["case"]
        label = f"{c['dtype']}  M={c['M']:,}  N={c['N']:,}  {mode_name[c['mode']]}"
        s = _stats_us(r)
        if s is not None:
            by_case[label][r["provider"]] = s
    rows = sorted(by_case.items(), key=lambda kv: -max(s[1] for s in kv[1].values()))
    _dot_plot(rows, providers, set(),
              "RMSNorm, stress cases", "median latency (µs), log scale",
              outdir / "rmsnorm_stress.png",
              "Dot = median of ≥100 iterations; bar = interquartile range. NVIDIA L40S.")


# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark JSON results.")
    parser.add_argument("--results", type=Path, default=BENCH_DIR / "results", help="JSON input dir")
    parser.add_argument("--outdir", type=Path, default=BENCH_DIR / "plots", help="PNG output dir")
    args = parser.parse_args()

    if not args.results.is_dir():
        raise SystemExit(f"results dir not found: {args.results}")

    records = load_all_records(args.results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for V in sorted({r["case"]["V"] for r in records if r["suite"] == "sampling"
                     and r["case"]["B"] in (1, 32, 256) and r["case"]["k"] in (1, 50, 500)}):
        if V in (32768, 131072):
            plot_sampling_core(records, args.outdir, V)
    plot_sampling_stress(records, args.outdir)
    for dtype in ("float16", "bfloat16"):
        plot_rmsnorm_core(records, args.outdir, dtype)
    plot_rmsnorm_stress(records, args.outdir)
    print(f"wrote plots to {args.outdir}")


if __name__ == "__main__":
    main()
