# FeCAM 困难分类数据补充 v59

沿用已固定的 FeCAM 时序特征适配代码及六值 alpha1 网格 .01/.1/.5/1/2/10。新增覆盖 PhonemeSpectra、LSST、Handwriting，两种表示mantis/summary，512维同一StableSemanticBasis、seed42/order17，第一阶段校准，后续仅更新已到达训练样本统计。共36条验证轨迹，8个CPU进程并行。无需保存真实样本，不受ER10缓存额度影响，累计类均值/协方差及逆矩阵单独计内存。

仅TRAIN/VAL，选最终VAL宏准确率最高的alpha（相同指标取较小alpha）；同时保留所有配置及平均阶段指标。不得将该适配当作原论文原数据复现，不直接拓展到回归。代码来源和MIT许可证见 third_party/fecam/PROVENANCE.md。与当前TMoT比较需核对输入hash和任务顺序一致。
