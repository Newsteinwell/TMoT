# v72 首任务时间留出初始化结果

辅助训练源码4dfd4d7eb88d7f69a78534f15f9b578555a7effc，核验与规则重放源码19a2cecbe65dfb2da2925b04de563c0f733c6e01。四个数据/表示组的首任务前缀辅助训练与核验完成，48条完整规则轨迹核验完成。Air为16训练/5内部校准，北京为128/32；外层VAL未参与初始化。

实际专家仍是v70使用全部已到达TRAIN的模型；辅助训练前缀与重拟合生产专家不同，校准误差只是选择代理。六值ridge最终按旧VAL选择，所有结果属于开发诊断，无TEST，不是独立盲测。

| 数据集 | 表示 | 方法 | 初始化后最终RMSE | 初始化后阶段平均 | v71阶段平均 | 初始化后分支序列 |
|---|---|---|---:|---:|---:|---|
| air_quality | mantis | ridge | 1.0341 | 0.8924 | 0.8271 | residual/append/append/append/append |
| air_quality | mantis | task | 0.9772 | 1.0036 | 0.8457 | residual/residual/residual/residual/append |
| air_quality | summary | ridge | 0.9076 | 0.9181 | 0.8141 | residual/residual/append/append/append |
| air_quality | summary | task | 0.8265 | 0.7576 | 0.8078 | none/none/none/none/none |
| beijing_pm25 | mantis | ridge | 27.6954 | 30.5503 | 34.7851 | residual/residual/residual/residual/residual |
| beijing_pm25 | mantis | task | 27.3419 | 29.1435 | 33.3061 | residual/residual/residual/residual/residual |
| beijing_pm25 | summary | ridge | 27.1269 | 28.5739 | 35.7021 | residual/residual/residual/residual/residual |
| beijing_pm25 | summary | task | 27.0923 | 29.0761 | 34.5885 | residual/residual/residual/residual/residual |

## 结论

北京四个方法/表示条件都从首任务起选择residual，恢复该单专家的完整轨迹。TMoT阶段平均由33.31/34.59恢复到29.14/29.08，但最终仍为27.34/27.09，没有超越已有强专家；同先验ridge仍为27.70/27.13，summary阶段平均28.57优于TMoT29.08。

Air+summary的TMoT一直选择none，阶段平均由0.8078改善至0.7576；这是恢复已有none轨迹，不是新收益。Air+Mantis却在前四个任务选择residual，阶段平均从0.8457退化到1.0036。两个Air ridge的阶段平均也退化。内部留出并没有形成可通用的初始化机制。

首任务初始化不足解释了北京的额外损失，但小样本校准、前缀模型与全任务重拟合模型差异，以及训练前误差与训练后目标不一致仍存在。不能根据这轮VAL结果单独让北京启用初始化、Air禁用后再声称统一算法。后续应优先检验选择信号的估计可靠性或重新设计学习目标，而非继续对同一VAL寻找分支规则。

## 核验与成本

核验前缀/后缀原始窗口不重叠；basis、输入参考缩放、目标均值/标准差仅由前缀重建；TMoT精确输入矩和目标交叉矩与前缀直接统计一致；所有内部校准预测由检查点精确复现。重放阶段从这些内部预测及v70训练前预测独立累计SSE、核对样本身份、选择索引和各阶段指标；源文件SHA不变。

增加首任务三个辅助神经专家的训练，生产阶段仍需维护三个专家，不能声称节省单模型预算。保留全部负面结果，不升级默认算法；一般多维时序持续学习、类增长与普通连续标签回归的总体范围不变。
