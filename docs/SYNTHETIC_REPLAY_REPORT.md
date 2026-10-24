# v52 非线性统计匹配合成回放结果

源码 be9ade9f1870488794b1d04629872452a50e74ca。36 次共享训练、216 条完整 VAL 轨迹全部完成，20 组并发。8 项针对性测试通过。保存全部结果，TEST 未读取，正式十数据集验收不变。

## 结论

本候选未达到普遍稳健优势，不替换 Gaussian 默认。随机非线性统计匹配的优化目标显著下降，但多数退化说明降低这个代理目标并不足以保证任务训练信息得到保留。该结果不能证明所有合成回放无效，也不能提前肯定其他匹配目标。

候选每分量保留 16 个合成代表，不存真实历史行；仍保留原统计量。分量数随到达增长，所以不是固定总内存方法。finite 与 matched 数组内存相同，后者增加优化计算。下表各列独立同六点 ridge 选择；ER 为同管线历史参考，未声称总内存与计算严格相同。

| 数据集/表示 | Gaussian | finite | matched | ER |
|---|---:|---:|---:|---:|
| air_quality/mantis | 0.98832 | 0.93869 | 0.96342 | 0.98626 |
| air_quality/summary | 0.83993 | 0.85040 | 0.82302 | 0.75155 |
| beijing_pm25/mantis | 46.14978 | 44.86581 | 45.05464 | 42.92421 |
| beijing_pm25/summary | 45.58701 | 45.04127 | 46.01196 | 39.28178 |
| energy/mantis | 63.71638 | 64.57920 | 66.39852 | 66.97470 |
| energy/summary | 66.54252 | 68.85011 | 67.71553 | 68.22038 |
| handwriting/mantis | 51.92308 | 53.84615 | 46.15385 | 48.07692 |
| handwriting/summary | 23.07692 | 15.38462 | 19.23077 | 30.76923 |
| lsst/mantis | 59.69511 | 59.55241 | 60.93753 | 60.03831 |
| lsst/summary | 29.36688 | 26.08207 | 29.74740 | 34.77763 |
| phonemespectra/mantis | 32.57919 | 31.82504 | 31.97587 | 33.63499 |
| phonemespectra/summary | 10.85973 | 8.14480 | 7.99397 | 10.40724 |

分类单位为宏准确率百分比，回归为 RMSE。LSST/Mantis 的 matched 最终 60.94%，高于旧 Gaussian 59.70%，但相对 ER 60.04% 不到 1 个百分点；其他强参考和多种子验证也没有因此被替代。

## 阶段平均与资源

- air_quality/mantis：对应选模配置的阶段平均 Gaussian/finite/matched = 0.77276/0.77115/0.77876。
- air_quality/summary：对应选模配置的阶段平均 Gaussian/finite/matched = 0.76540/0.76553/0.79453。
- beijing_pm25/mantis：对应选模配置的阶段平均 Gaussian/finite/matched = 52.68826/52.95507/53.33173。
- beijing_pm25/summary：对应选模配置的阶段平均 Gaussian/finite/matched = 56.36226/59.09503/58.26373。
- energy/mantis：对应选模配置的阶段平均 Gaussian/finite/matched = 71.54000/71.07714/71.67854。
- energy/summary：对应选模配置的阶段平均 Gaussian/finite/matched = 76.43804/76.57681/78.87286。
- handwriting/mantis：对应选模配置的阶段平均 Gaussian/finite/matched = 46.79746/44.61570/44.87182。
- handwriting/summary：对应选模配置的阶段平均 Gaussian/finite/matched = 20.56844/19.87432/18.78740。
- lsst/mantis：对应选模配置的阶段平均 Gaussian/finite/matched = 69.98565/68.42140/70.77351。
- lsst/summary：对应选模配置的阶段平均 Gaussian/finite/matched = 46.02248/42.06877/44.18025。
- phonemespectra/mantis：对应选模配置的阶段平均 Gaussian/finite/matched = 48.93676/49.36006/48.38623。
- phonemespectra/summary：对应选模配置的阶段平均 Gaussian/finite/matched = 17.30336/17.13111/16.88314。
- gaussian：共享组耗时中位数 21.33s，统计数组 5.76—86.93MiB。时间受并发竞争影响，不能当作独占 GPU 基准；数组不含 Python/RNG 对象。
- finite：共享组耗时中位数 20.92s，统计数组 6.17—91.72MiB。时间受并发竞争影响，不能当作独占 GPU 基准；数组不含 Python/RNG 对象。
- matched：共享组耗时中位数 22.42s，统计数组 6.17—91.72MiB。时间受并发竞争影响，不能当作独占 GPU 基准；数组不含 Python/RNG 对象。

## 独立审计

重建 12 个 matched 状态的 552 个真实 TRAIN 分区，并以 NumPy 独立复算投影、ReLU、均值/Gram 和回归交叉矩目标。最终/初始目标比值中位数 0.01439，最大值 0.04684；所有分量目标不增。

216 条指标、输入哈希、覆盖检查通过；36 个共享状态、独立读出求解通过（最大绝对误差 9.16e-11）。72 个 Gaussian 控制逐阶段精确复现 v48；144 个非控制轨迹与 Gaussian 的主 RNG、分区、原矩、训练预算与第一阶段预测一致。finite/matched 的代表抽样 RNG 与数组内存一致，所有代表形状/有限性检查通过，保存状态无 observed 真实行字段。

独立目标审计初版在核对回归分量均值前过早转为 float32，造成约 1e-8 差异；修正审计精度顺序后通过，未修改实验或放宽均值精确检查。优化目标按实现原有 float32 计算，并允许 NumPy/PyTorch 运算顺序的正常误差。

## 后续决策

保留本轮失败与局部改善证据，下一轮应检验任务相关的已学习特征或梯度匹配，而非直接增加随机投影数量或基于 VAL 逐数据集挑选方案。首先维持相同代表数量、初始化与优化成本，对照随机特征与任务相关目标，再决定是否值得多种子扩大实验。尚未证明此后续候选有效。
