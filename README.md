# TOF-Net

A learned **Time-of-Flight (TOF) predictor** bolted onto the [G-FOLD](https://github.com/samutoljamo/g-fold) solver. Offline Monte-Carlo sampling produces a "state/constraints → optimal TOF" dataset; a dual-head MLP (feasibility classification + TOF regression) is trained on it. At runtime the predictor skips the solver's internal TOF search, enabling high-frequency MPC powered-descent guidance.

This repository implements all three modules: **data generation**, **training**, and **inference**.

## Directory layout

```
GFOLD-solver/
├── common/                 # Shared contract (used by all three modules)
│   ├── config.py           # Feature schema + sampling ranges + min-max bounds + constants (no torch)
│   └── model.py            # Dual-head MLP architecture (shared by training/inference, needs torch)
├── generation/             # Module 1: data generation
│   ├── sampling.py         # Causally-consistent sampling (pure functions)
│   ├── gfold_glue.py       # Feature vector -> gfold.Config + solve helpers
│   └── generate.py         # Multiprocess generation + rich TUI + shard resume + JSONL logging
├── training/               # Module 2: training
│   ├── dataset.py          # Reads shards + normalization + stratified split (GPU preload support)
│   ├── train.py            # BCE + λ·MSE training, PR threshold selection, checkpointing
│   └── evaluate.py         # Predicted-TF -> fixed-TF re-solve: success rate + fuel suboptimality
├── inference/              # Module 3: inference
│   └── predictor.py        # Loads a checkpoint -> (p_feasible, tf)
├── visualization/          # Interactive (plotly) HTML reports
│   ├── trajectory.py       # 3D trajectory + time-series panels
│   ├── dataset_eda.py      # Feasibility / mean-TF heatmaps + histograms
│   ├── training.py         # Training curves from train_metrics.csv
│   └── evaluate.py         # Prediction scatter, suboptimality, confusion matrix, PR curve
├── scripts/
│   ├── run_generate.py     # PyInstaller entry point
│   └── build_exe.ps1       # Build script for the data-generator exe
├── docs/                   # Design documents
├── pyproject.toml          # Package metadata (installable via pip install -e .)
└── .gitignore
```

## Installation

Environment: miniforge env `KRPC` (Python 3.12).

Install the package in editable mode so `common`, `generation`, `training`, `inference` and `visualization` are importable from any directory:

```bash
python -m pip install -e .            # base: numpy + gfold
python -m pip install -e ".[all]"     # + rich, psutil, torch, scikit-learn, plotly
```

Or install the dependencies manually:

```bash
python -m pip install gfold numpy rich psutil plotly
# torch via the CPU index (training/inference only; the data-generator exe does not need it)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> After `pip install -e .` you can run scripts as modules (`python -m inference.predictor`) or as plain files (`python inference/predictor.py`) from anywhere — the package paths resolve regardless of the working directory.

## Conventions (the shared contract — do not change casually)

### Coordinate frame

- **z-up**; gravity along −z, fixed at Kerbin surface gravity `g = 9.81 m/s²` (`GRAVITY = [0,0,-9.81]`)
- Target landing point is the **origin**, target velocity `[0,0,0]`
- Positions and velocities are relative to the target
- Fixed discretization `n = 100`; `max_velocity = 600` (ensures the initial state never violates the SOC velocity constraint)

### Feature schema (14-D, order is the contract)

```
[x, y, z, vx, vy, vz, dry_mass, fuel, real_max_thrust,
 min_thrust_pct, max_thrust_pct, fuel_consumption,
 glide_slope_angle_deg, max_angle_deg]
```

### Normalization

Min-max to `[0,1]` using `NORM_BOUNDS` in `common/config.py` (priors, not dataset statistics). They are written into `meta.json` at generation time and read identically by training and inference so the three stages never drift.

### Causal sampling chain (physically self-consistent)

```
Mass chain:      dry_mass -> dry_frac (= dry/wet_full) -> wet_full
                 -> remaining_frac -> fuel, wet_mass
Thrust chain:    twr_max -> real_max_thrust = twr_max * wet_mass * g
                 (min_thrust_pct < max_thrust_pct always; effective TWR = twr_max * max_pct > 1)
Isp chain:       isp -> fuel_consumption = 1 / (isp * 9.80665)
```

Full ranges live in `RANGES` in `common/config.py`.

### Two gfold API facts (see docs/gfold_Python_API文档.md)

1. **Nested-field assignment silently does nothing** — you must construct-inject or replace whole nested objects; `cfg.solver.time_of_flight = ...` is a no-op.
2. **The optimal TOF is not exposed directly** — recover it as `tf = time_points[-1] * n / (n - 1)`.

## Dataset format

Generation writes to `data/` (shards + metadata) and `logs/` (per-sample JSONL).

### `data/shard_<start>.npz`

Each shard holds 10,000 samples by default (`--shard-size`), with fields:

| Array | Shape | dtype | Description |
|---|---|---|---|
| `X` | (m, 14) | float64 | Raw features (unnormalized), columns per schema |
| `y_feasible` | (m,) | int8 | Feasibility label 0/1 |
| `y_tf` | (m,) | float64 | Optimal TOF (s); NaN for infeasible samples |
| `objective` | (m,) | float64 | Objective = ln(final mass); NaN if infeasible |
| `final_mass` | (m,) | float64 | Final mass (kg); NaN if infeasible |
| `fuel_used` | (m,) | float64 | Fuel consumed (kg); NaN if infeasible |
| `solve_ms` | (m,) | float64 | Solver time (ms) |
| `index` | (m,) | int64 | Global sample index (deterministic under a fixed seed; in-shard order is arbitrary, key on `index`) |

### `data/meta.json` — contract snapshot

Records `schema_version`, `features` (order), `norm_bounds`, `ranges`, `g`, `n`, etc. Training and inference must read this file rather than hard-coding.

### `data/manifest.json` — resume state

```json
{"n_done": 100000, "shards": ["shard_00000000.npz", ...]}
```

### `logs/samples.jsonl` — per-sample provenance log

One JSON object per line: `index`, `features`, `feasible`, `tf`, `objective`, `final_mass`, `fuel_used`, `status`, `solve_ms`. For infeasible samples `status` is the gfold exception message (usually `infeasible: no feasible time-of-flight in [...]`).

## Generation (module 1)

Run from the project root:

```bash
# Generate 100k samples (default: all logical cores, seed=0, shard=10k)
python -m generation.generate --n-samples 100000

# Pilot batch: run 2000 samples to check feasibility rate (writes to data_pilot/)
python -m generation.generate --pilot 2000

# Append: grow the same dataset from 100k to 200k (same seed/outdir auto-resumes)
python -m generation.generate --n-samples 200000

# Explicit cores / seed / shard size
python -m generation.generate --n-samples 100000 --workers 64 --seed 1 --shard-size 20000
```

### CLI arguments

| Arg | Default | Description |
|---|---|---|
| `--n-samples` | 100000 | **Total** dataset target; auto-resumes the remainder if a manifest exists |
| `--pilot N` | 0 | Pilot batch: run N samples to measure feasibility, writes to `*_pilot` dir |
| `--workers` | `os.cpu_count()` | Parallel process count (gfold does not release the GIL, so multiprocessing is required) |
| `--seed` | 0 | Sample i uses `seed+i`; a fixed seed reproduces the whole dataset |
| `--shard-size` | 10000 | Samples per `.npz` |
| `--outdir` | data | Output dir for shards + manifest + meta |
| `--logdir` | logs | JSONL log dir |
| `--logfile` | samples.jsonl | Log filename |
| `--no-write` | false | Don't write shards (calibration) |
| `--no-tui` | false | Disable the TUI |

### Append / resume semantics

- `--n-samples` always means the **dataset total**. A first `--n-samples 100000` produces indices 0–99999; re-running `--n-samples 200000` (same `--outdir`/`--logdir`/`--seed`) detects 100000 already done in the manifest and continues from index 100000, appending to the same shards and log.
- With the same `--seed`, resuming does not affect reproducibility (each index has a deterministic sampling seed).
- Crash-safe: shards are written to `.tmp.npz` and atomically renamed, and the manifest is updated only after a shard is fully written; re-running after an interrupt resumes from the last completed shard boundary.

### TUI

On an interactive terminal a rich full-screen TUI starts automatically: top progress (total/feasibility/rate/shards/elapsed ETA), a per-core CPU block map with the active-worker count, and a scrolling worker success/failure log. Non-TTY output (e.g. server SSH) degrades to one progress line every 5 seconds; `--no-tui` forces it off. `Ctrl+C` exits gracefully (flushes the current shard + writes the manifest).

## Packaging the exe (for a server)

The generator does not depend on torch, so the exe only bundles gfold/numpy/rich/psutil.

```powershell
# PowerShell (KRPC env)
powershell -File scripts\build_exe.ps1
# or manually:
pyinstaller --name tofnet_generate --onedir --clean --noconfirm `
    --paths . --collect-all gfold --hidden-import generation.generate `
    scripts\run_generate.py
```

The output is `dist/tofnet_generate/`. Copy the **whole folder** to the server, then:

```bash
./tofnet_generate.exe --n-samples 100000 --workers <server cores>
```

- Use `--onedir` (not onefile): with multiprocessing spawn, child processes load the already-extracted binaries directly, avoiding onefile's per-process re-extraction cost.
- No Python/gfold/torch environment is needed on the server; everything is adjustable via command-line args.

## Training (module 2)

```bash
# Full training (auto-uses GPU and preloads data into VRAM when available)
python -m training.train

# Common options
python -m training.train --epochs 50 --batch 4096 --lr 3e-4 --device auto
```

- Loss `BCE(feasibility) + λ·MSE(TF, feasible samples only)`; TF is z-scored, inputs are min-max normalized (same `meta.json`).
- Stratified 80/10/10 split by feasibility; the validation PR curve selects the threshold (`--beta` sets Fβ, β<1 favors precision).
- Outputs `models/tofnet.pt` (weights + self-contained contract) and `models/tofnet.json` (human-readable).

## Inference (module 3)

```python
from inference.predictor import TOFNetPredictor
pred = TOFNetPredictor("models/tofnet.pt")     # resident in memory, loaded once
p, tf = pred.predict(features)                  # features: 14-D raw physics (FEATURES order)
feasible, p, tf = pred.decide(features)         # decision at the trained threshold
```

```bash
python -m inference.predictor --ckpt models/tofnet.pt   # demo
```

- Single sample returns `(float, float)`; a `(B,14)` batch returns tensors; `predict_from_dict(dict)` picks values by feature name.
- Online, use `decide`: `p >= threshold` means feasible; below the threshold, fall back to a full TOF search.

### Runtime note: run the predictor on CPU

At runtime the GPU is typically occupied by the game/renderer, and for a 27k-parameter MLP the CPU forward is actually **faster** than the GPU (kernel launch overhead dominates). Measured single-sample prediction: **~0.4 ms median on CPU vs ~1.5 ms on GPU**.

```python
import torch
torch.set_num_threads(1)                                   # single-sample inference: cut thread-pool jitter
pred = TOFNetPredictor("models/tofnet.pt", device="cpu")   # resident, loaded once
```

### ONNX export (optional)

```bash
python -m inference.onnx --ckpt models/tofnet.pt   # exports + validates + latency compare
```

Exports a self-contained `models/tofnet.onnx` — input is the 14-D raw feature vector, outputs are `p_feasible` and `tf` (seconds), with normalization / sigmoid / de-standardization baked into the graph. Run it with onnxruntime (no torch needed):

```python
from inference.onnx import OnnxPredictor
pred = OnnxPredictor("models/tofnet.onnx")
p, tf = pred.predict(features)
```

Measured single-sample forward: ~1.1 µs (onnxruntime) vs ~4.9 µs (torch CPU) — ~4.3× faster and lower-jitter. This only trims the prediction step; the end-to-end time is still dominated by the gfold solve (~14 ms), so the total speedup stays ~32×.

## Benchmark (G-FOLD search vs TOF-Net + fixed-TF solve)

```bash
python -m training.benchmark --n 300 --device cpu
```

On the same held-out feasible samples (n=300, predictor on CPU):

| Path | median | p90 |
|---|---|---|
| A. G-FOLD internal TOF search | 444 ms | 523 ms |
| B. TOF-Net fixed-TF solve | 14.0 ms | 17.8 ms |
| └ prediction forward | 0.4 ms | 0.5 ms |
| └ end-to-end (predict + solve) | 14.5 ms | 18.5 ms |

**Speedup: ~32×** (median), fixed-solve success rate **96.7%**. The end-to-end median of ~14.5 ms fits the 60 Hz budget (~16.7 ms); the ~3% of samples whose predicted TF fails fall back to a full search.

## Evaluation (downstream value)

```bash
python -m training.evaluate --n-feasible 1000 --n-infeasible 1000
```

For held-out feasible samples, re-solves gfold at the predicted TOF and reports solve success rate + fuel suboptimality, compared against a "constant TOF" baseline and an oracle (optimal-TF label); also reports the classification cost at the threshold.

## Visualization (interactive)

All output is self-contained HTML (plotly.js embedded) written to `figures/`.

```bash
python -m visualization.trajectory --random --seed 1      # 3D trajectory + time series
python -m visualization.dataset_eda --surface             # feasibility/mean-TF heatmaps (+3D surfaces)
python -m visualization.training                          # training curves
python -m visualization.evaluate --n-feasible 500         # evaluation plots
```

## Roadmap

- Online MPC closed loop (kRPC + KSP) — this lives in the KSP-Auto-Landing project; this repository only provides the estimator.
