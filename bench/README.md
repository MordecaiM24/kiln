# Benchmark harness

## One-command reproduction (remote L40S)

From your Mac, sync the repo and run the full matrix on the GPU host:

```bash
rsync -avz --exclude '.venv' --exclude '__pycache__' \
  -e "ssh -i ~/.ssh/vcl" ~/kiln/ mgmenges@152.7.176.102:~/kiln/

ssh -i ~/.ssh/vcl mgmenges@152.7.176.102 \
  'cd ~/kiln && uv run python bench/run_bench.py --suite all && uv run python bench/plot.py'
```

Quick harness check (2 tiny cases per suite):

```bash
uv run python bench/run_bench.py --suite all --smoke
```

Plots read only committed JSON under `bench/results/`; re-run `bench/plot.py` after syncing new results.

## Timing protocol

1. Steady-state latency uses `triton.testing.do_bench` with `warmup=25`, `rep=200` (bumped to `rep=500` if fewer than 100 samples).
2. Every record stores all raw per-iteration times (`raw_ms`); median and IQR (Q3−Q1) are derived from those replicates.
3. `torch.compile` providers report `compile_time_s` separately (wall clock + `torch.cuda.synchronize`) before `do_bench`.
4. Failed or unavailable configs are written with `status: "failed"` or `"unsupported"` and an error string — never omitted.
5. Plots regenerate purely from the JSON via `bench/plot.py`; commit raw JSON in `bench/results/` alongside code changes.
