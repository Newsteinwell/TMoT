# FeCAM classifier reference

Official source: https://github.com/dipamgoswami/FeCAM
Pinned commit: e33f39d112ff2d2a2df2e68c490af579a50edd31
Reference: FeCAM_vit_cifar100.py (L2 features, unbiased per-class covariance,
identity shrinkage alpha1=1, alpha2=0, covariance-to-correlation normalization,
pseudoinverse Mahalanobis classification). License copied alongside this file.

TMoT adaptation: use the same frozen 512-dimensional StableSemanticBasis as
other evaluated heads; retain mergeable statistics for recurring classes;
cache precision matrices; evaluate row-wise distances without an N-by-N matrix.
No Tukey transform is used (inputs have signed coordinates; official pretrained
ViT protocol also omits it). Guard zero variance/singleton classes explicitly.
This is a FeCAM head adaptation to time-series features, not a reproduction of
its original image backbone/benchmark results. alpha1 receives six validation
settings, including the official value 1. alpha2 stays 0.
