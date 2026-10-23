# Pilot protocol (2026-10-24)

## Classification

Three data sources × six methods × two class orders × two initialization seeds = 72 complete trajectories. Class orders: 17,29; initializations: 42,43. Two new classes per stage; odd class counts use three in the first stage. Inference competes over all observed labels, no task ID.

- Common frozen MantisV2: revision `8f6ca35cb54ab14b120618943c6fca5ddf5a76a6`, intermediate layer index 2, combined token, sample-local per-channel zscore, channel-ordered embeddings plus channel mean/SD.
- Shared basis: first-stage-only feature standardization, seeded random projection plus ReLU to 256, seed1701. This is shared across methods for the same stream order; it is not fitted using all classes.
- TMoT / FT / ER: 64-dimensional affine adapter. 30 epochs per stage, batch128, AdamW lr0.001, wd0.0001. TMoT current CE + 0.5 SupCon + 1.0 historical squared loss. Prespecified, not tuned on pilot test results.
- Readout: TMoT/analytic variants ridge0.01; SLDA-style covariance 0.1 isotropic shrinkage.
- Feature ER: 4MiB feature/label/priority reservoir budget. Other methods' actual state sizes are reported, **not claimed byte-matched** at this stage.
- Baseline names are mechanisms, not official reproductions: `ridge` is cumulative class-balanced ridge, `ranpac` is a RanPAC-style fixed512 random-ReLU plus ridge; no foundation first-session fine-tuning. `slda` keeps per-class full moments for consistency and is not the most memory-efficient possible SLDA implementation.
- Primary pilot score: average stagewise macro class accuracy. Also final macro accuracy, micro accuracy, forgetting (can be negative), runtime, peak memory and retained bytes.

## Regression

Appliances Energy: 5 consecutive temporal domains. Each domain split 60/20/20 on the raw timeline; build nonoverlapping 24-step input / 6-step future targets within each split with stride30. Input uses historical appliances, lights and indoor T/RH readings, omitting interpolated weather and random rv columns. Target is the mean of next six `Appliances` entries in Wh per interval, not total six-interval energy.

A fixed 32-step resampling of sample-normalized channels plus mean/SD provides the initial regression encoding. This is a real-data engineering pilot with a lightweight encoder, **not an encoder-matched comparison with classification**. Mantis regression is an additional planned run. Five methods × two seeds = 10 trajectories. No random reordering of time domains. Measure stagewise and final RMSE/MAE in target units. Evaluate retained old temporal distributions as well as newly learned domains; no claim of general zero-shot future forecasting.

## Isolation and reproducibility

Preparation records source ZIP hashes, sample IDs, shape and split manifests. Learners consume only current training records plus permitted aggregates/replay. Evaluator caches live outside the learner. Future classes are selected by the stream generator, never supplied to the learner constructor. No validation/test metrics influence optimization, early stopping or hyperparameters in this pilot.

Each run saves per-stage predictions, metrics, config, feature hashes, source version, system measurements and an aggregate checkpoint. GPU and CPU timings include the readout within stage fit. More granular profiling and byte-matched comparisons are required before formal conclusions. Initial test outcomes are exploratory; method changes after observing them require new locked confirmation experiments.
