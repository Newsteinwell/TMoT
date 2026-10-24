# 数据集扩展预登记（新增模型结果读取前）

用户已明确同意加入更难数据集并保留全部现有结果。新增固定候选为PhonemeSpectra、LSST和Handwriting，不按TMoT初步成绩择取。选择理由分别为39类语音、多领域且类别不均衡的14类天文光变、150个官方训练样本覆盖26类的腕部动作。它们构成结构上更有挑战的压力集；尚未得到本项目成绩，不能预先断言任何一种方法在其上更弱或更强。

## 来源及资格检查

- PhonemeSpectra：官方表3315训练/3353测试、11通道、217长度、39类，频带随时间形成谱序列。[官方来源](https://timeseriesclassification.com/description.php?Dataset=PhonemeSpectra)
- LSST：2459/2466、6通道、长度36、14类，模拟天文光变的归档快照；不把填充/截断后的索引描述为完整原始不规则时间戳。[官方来源](https://timeseriesclassification.com/description.php?Dataset=LSST)
- Handwriting：150/850、长度152、26类。官网正文说6通道而表说3通道，实际TRAIN归档确认3通道，以文件为准。[官方来源](https://timeseriesclassification.com/description.php?Dataset=Handwriting)

资格检查仅读取TRAIN：固定矩形数值输入、全部有限、每类至少2例，以便保留训练和验证。记录归档SHA256、形状、逐类样本数；不计算模型成绩。任何结构不符合必须停止并记录原因，不自动替换成效果更好的数据。服务器下载403时允许同一官方URL经本机下载后上传，内容哈希应一致。

## 扩展后的固定集合与验收

结构核验通过后，集合固定为原7个（NATOPS、Epilepsy、RacketSports、PenDigits、ArticularyWordRecognition、Beijing PM2.5、Air Quality）加上述3个，共10个，8分类/2回归。原7个结果和原4个开发集全部保留，不能删除已达天花板的数据。原7集标准仍单独报告，不追溯宣称其达标。

扩展集合的多数阈值为至少6/10个明确改善，其中至少5/8个分类、至少1/2个回归；两任务族整体方向均须改善。单集明确增益沿用分类最终宏准确率至少+1个百分点、回归RMSE至少下降5%，均相对该集预先声明强基线中的最强者，阶段平均表现不得反向显著恶化。沿用5种子×3分类类序、回归5种子固定时间顺序，逐类序配对报告；数据集是跨数据集结论的单位，不能把运行数当数据集数。该标准是研发验收目标，不是当前已达到的结论。

本轮只做VAL筛选，正式TEST前另锁定最终配置、各基线预算和统计比较细节，不能从测试分数回调模型。新回归候选LFMC仍处于来源/域划分核验阶段，本次不计入分母；未来纳入必须事前单独修订。

## 不变的划分与首轮矩阵

沿用官方TEST，官方TRAIN内按类别固定seed1701划20%向下取整且至少1例作为VAL；不重采样以扩大验证集。TRAIN最小类计数：LSST为7、Handwriting为2，因此Handwriting部分新类只有一个训练例、VAL很小，单次VAL差不能当稳健证据。没有参与者ID，不声称主体独立。

保留已有预处理，冻结Mantis与无预训练summary两种表示；本轮只提取TRAIN/VAL特征。首次筛选固定seed42/order17，分类每阶段2类，奇数类别首阶段3类；三数据×两表示×（累计ridge、local正则化创新、ER正则化创新）×六个ridge（.001/.01/.1/1/10/100），共108条。后两者保持既有网络/步数/回放，输入校准暂固定first，避免新表示效应和新校准改动混杂。每方法同6点选择预算；记录实际内存与时间，不能把ER 4MiB与按类别增长的矩内存声称为等字节。后续强基线仍包括原迁移版与SLDA；不能只相对这轮两个对照给出最终胜出结论。

保持训练与测试严格分离；Mantis预训练重叠仍未知，summary对照不能使该不确定性自动消失。PenDigits仍保留但属于空间轨迹；新增EthanolConcentration不纳入，因为其轴为波长而非时间，避免扩大时间序列证据的解释范围。

## 归档资格检查完成

三个官方归档全部通过TRAIN-only有限性、矩形结构、类别样本数核验。固定划分后的train/val/test分别为PhonemeSpectra 2652/663/3353、LSST 1973/486/2466、Handwriting 122/28/850；全部十个数据集因此正式进入上述固定验收集合。Handwriting三通道差异已按文件解决。归档、划分SHA256将随inventory和manifest保存；官方TEST仅按固定规则存档，未计算模型评分。

执行方式：`python -m scripts.inspect_extension_archives ROOT`、`python -m scripts.prepare_extension --root ROOT`。使用模块入口使共享预处理导入路径明确。流水线先2并发提取6组TRAIN/VAL缓存，成功后自动启动16并发的108条VAL实验；每阶段启动前要求剩余空间至少15GiB。源码、数据资格与实验方案在新增成绩前锁定。
