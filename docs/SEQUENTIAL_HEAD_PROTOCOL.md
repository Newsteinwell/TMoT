# v73 顺序监督头与输出蒸馏对照

目的：补齐匹配warm ER10训练流程的无回放FT与蒸馏对照，区分历史回放与读出贡献。不是TMoT新模块收益实验。

固定原开发TRAIN/VAL、seed42/order17，数据Handwriting、LSST、PhonemeSpectra、Air Quality、Beijing PM2.5、Energy；Mantis/summary两表示。稳定512维基底与首任务输入/目标校准保持v58；128宽两层网络、30epochs、AdamW、batch256相同，阶段继承网络与监督头，重置优化器。分类所有已见类联合CE，当前类平衡；回归MSE。FT strength=0；蒸馏strength=.1/1/10，事先固定，保留全部结果。每阶段仅用当前TRAIN。

分类使用LwF思想：前一模型对当前输入产生旧类软目标，旧类子空间KL，T=2、乘T²。这是单头类增长适配，不是原文多任务实验复现。回归为相同连续输出MSE蒸馏，不称原版LwF。旧模型只读，教师预测不跨阶段缓存，无历史样本回放。出处：https://arxiv.org/abs/1606.09282 。

48条完整轨迹、8并发、仅最终小模型与各阶段预测，无大型统计检查点。验证所有首阶段预测精确匹配v58 warm ER10监督头；重算各阶段指标，核对输入hash与最终检查点预测。最终VAL选择蒸馏强度仅为开发比较，不使用TEST、不称盲测。不把这轮较弱基线当作替代FeCAM/ridge强基线的依据。
