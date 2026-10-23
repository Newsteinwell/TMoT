# Design: one framework, classification and regression

The project objective is **general multivariate time-series continual learning**, not merely Wi-Fi sensing or fixed-vocabulary classification. The implementation progresses through classification, class growth and regression. Evidence for one setting must not be silently generalized to an untested setting.

Let a sample-local frozen encoder and first-stage-only normalization form phi(x). Adapt h=A phi+b. For each retained group or class, moments obey mu_h=A mu+b and Sigma_h=A Sigma A^T. This identity is independent of input modality and distributional Gaussianity. It preserves moments, not classification performance.

For z=[phi,1], store G=E[zz^T], K=E[zy^T], V=E[yy^T]. For an affine output Fz:

`E[||Fz-y||²] = tr(F G F^T) - 2 <F,K^T> + tr(V)`.

## Class growth

Only observed classes enter the learner. Each new class appends a head row. For historical examples, the targets in all new rows are zero. Thus the exact historical quadratic risk constrains newly introduced competitors as well as old outputs. Per-class raw moments allow class-balanced risk to be reconstructed for any currently seen label set. No old samples and no future labels are required.

The initial implementation uses uncentered one-hot targets. Extending WiMoT's centered targets is another valid option but requires rebuilding class cross moments when C changes; padding the old centered matrix is not generally correct. Classifier dimensions and label mappings are explicit, with a test using nonconsecutive labels and recurring classes.

## Regression

Replace one-hot labels by scalar/vector continuous targets and retain G,K,V. Fixed-dimension output regression has the same exact quadratic identity. The pilot uses first-domain target normalization frozen over the stream, and reports errors back in physical target units. Classification is not reduced to numeric label regression.

No finite first/second-moment summary determines arbitrary nonlinear losses. Extending to other task objectives may need approximations, fixed feature lifts, richer statistics, or alternative heads. This is a research boundary to investigate, not a change to the general research objective.

## Initial TMoT pilot vs WiMoT

The initial pilot intentionally tests the moment mechanism in a minimal common framework: frozen pretrained per-channel features; shared first-stage standardization and random ReLU basis; 256→64 affine adapter; current CE+SupCon for classification / MSE for regression; historical squared risk; analytic ridge readout refitted from transported moments. It does not yet include the original WiMoT semantic stem, first-domain PCA, Student-t calibration or OSCL pair optimization. No claim of faithful original accuracy reproduction is made.

The ridge head is refitted after adaptation. Therefore low historical head loss does not automatically imply high final classification accuracy. `no_hist` and `frozen_adapter` are implemented to test whether representation adaptation is beneficial. An unconstrained full-rank invertible adapter may leave an unregularized full-information linear readout invariant; the bottleneck and regularization make this an empirical representation-selection question, not a free gain.

A broad paper will require multiple datasets/protocols, real domain drift, strong recent baselines and failures—not merely renaming WiMoT or proving the standard affine identity again.
