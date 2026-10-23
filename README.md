# TMoT — Time-series Moment Transport

**Research objective: a general continual-learning algorithm for multivariate time series.**
We investigate class-incremental classification, domain changes and continuous-target regression in a unified moment-transport framework. These are research goals, not a claim that all time-series tasks have already been validated.

TMoT builds on the ideas of [WiMoT](https://github.com/Newsteinwell/WiMoT) and [OSCL](https://openreview.net/forum?id=Vf6WcUDnY7c). This repository starts a new experimental implementation; it is **not a renamed reproduction** of the full WiMoT pipeline. In particular, the initial TMoT pilot uses a common fixed feature basis, an adaptive affine bottleneck and an analytic ridge readout instead of the original dual stem and OSCL classifier.

## Implemented

- Class growth without knowing future labels: sparse class moments and dynamically expanded outputs.
- Exact historical squared-risk evaluation including new output rows on old classes.
- Scalar / vector regression using input, input-target and target second moments.
- Exact affine transport of the stored moments in a fixed basis.
- Pilot baselines: fine-tuning, feature replay, streaming LDA-style head, cumulative ridge and RanPAC-style random nonlinear features. These are documented implementations, **not yet faithful official ACIL/RanPAC benchmark reproductions**.
- Subject-disjoint HAR and 19-class DailySports, official UEA UWave splits, and chronological Appliances Energy regression preparation.
- Frozen per-channel MantisV2 extraction with ordered concatenation; no minimum-channel constraint.
- Bounded resumable experiment queue, stagewise predictions, hashes, timings and persistent-state byte accounting.

## Current limitations

The basis stays fixed after first-stage normalization. Adaptation is affine. Classification currently pools domains within each class and uses a global readout; sparse domain-conditioned OSCL and fixed-byte memory compression are planned. Regression currently has a fixed continuous output dimension. There is no assertion of zero forgetting, formal privacy, arbitrary nonlinear representational adaptation or established performance superiority.

## Setup

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -e '.[features,test]'
.venv/bin/python -m pytest -q
```

The first server run uses Python 3.12, PyTorch 2.8.0+cu128 and one 96GB RTX PRO 6000 Blackwell Server Edition. Dataset paths and credentials are intentionally absent from the repository. Set `TMOT_ROOT` to an external artifact directory with `raw`, `prepared`, `features`, `models`, `runs`, and `logs` subdirectories.

```bash
.venv/bin/python scripts/prepare_data.py --root "$TMOT_ROOT" --datasets har uwave dailysports energy
.venv/bin/python scripts/extract_features.py --root "$TMOT_ROOT" --datasets har uwave dailysports --weights "$TMOT_ROOT/models/MantisV2"
.venv/bin/python scripts/extract_features.py --root "$TMOT_ROOT" --datasets energy --encoder summary
.venv/bin/python scripts/run_queue.py --root "$TMOT_ROOT" --config configs/pilot_cil.json --name pilot_cil_v1 --workers 2
.venv/bin/python scripts/run_queue.py --root "$TMOT_ROOT" --config configs/pilot_regression.json --name pilot_regression_v1 --workers 2
```

The queue refuses to overwrite incomplete jobs, does not auto-retry failures, and has a four-hour per-trajectory timeout. Output folders are immutable per run name. Run-specific config and source hashes must accompany all results. Validation data are prepared but the initial pilot uses prespecified hyperparameters, without validation or test-driven selection. Public pilot test results become exploratory evidence; future confirmatory claims need an independent locked protocol.

See [protocol](docs/PROTOCOL.md), [mathematical design](docs/DESIGN.md), and [data sources](docs/DATA.md). Raw datasets, feature banks and training checkpoints are excluded from Git; compact aggregate results may be published under `results/`.

## Pilot results

See [execution status](docs/STATUS_2026-10-24.md) and [initial regression records](results/pilot_regression_v1/summary.csv). The first regression pilot does not outperform simple baselines; negative results are retained. All116 pilot trajectories have completed, including mechanism controls. See the [full pilot report](docs/PILOT_REPORT_2026-10-24.md). These are exploratory results, not paper-ready validation.
