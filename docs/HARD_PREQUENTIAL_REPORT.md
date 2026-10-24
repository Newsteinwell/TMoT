# v71 固定训练前误差下的硬选择

源码9a80d09bd3bb7578c0514e16f7a5ea47c0537e20；两项测试通过，48条完整轨迹核验完成。不重复训练，使用v70同一专家预测和训练前累计SSE；第一任务等权，之后选累计误差最小分支，同分按none/append/residual顺序。无VAL标签参与分支选择，无TEST。

六值ridge仍按旧开发VAL选取最终最优，非新盲测；不新增独立样本。数值为RMSE，越低越好。

| 数据集 | 表示 | 方法 | 硬选择最终 | 软加权最终 | 硬选择阶段平均 | 软加权阶段平均 | 硬选择分支轨迹 |
|---|---|---|---:|---:|---:|---:|---|
| air_quality | mantis | ridge | 1.0341 | 1.0413 | 0.8271 | 0.8292 | uniform/append/append/append/append |
| air_quality | mantis | task | 0.9772 | 0.9997 | 0.8457 | 0.7706 | uniform/none/residual/residual/append |
| air_quality | summary | ridge | 0.9076 | 0.9140 | 0.8141 | 0.7534 | uniform/append/append/append/append |
| air_quality | summary | task | 0.8265 | 0.8436 | 0.8078 | 0.7607 | uniform/none/residual/none/none |
| beijing_pm25 | mantis | ridge | 27.6954 | 31.1723 | 34.7851 | 38.0405 | uniform/residual/residual/residual/residual |
| beijing_pm25 | mantis | task | 27.3419 | 31.3105 | 33.3061 | 37.1404 | uniform/residual/residual/residual/residual |
| beijing_pm25 | summary | ridge | 27.1269 | 32.9517 | 35.7021 | 42.0345 | uniform/residual/residual/residual/residual |
| beijing_pm25 | summary | task | 27.0923 | 31.3538 | 34.5885 | 39.5233 | uniform/residual/residual/residual/residual |

## 可解释结论

北京从第二任务起均选择residual，最终恢复到该单专家的27.34/27.09（TMoT）和27.70/27.13（ridge）。这支持“平滑混合稀释强分支”的解释；最终恢复已有水平不算超越强基线。

北京阶段平均仍明显弱于固定残差专家，主要来自第一任务尚无训练前损失时采用等权：TMoT Mantis 33.31对29.14，summary34.59对29.08。首任务决定与后续训练前证据不同，不能用后续数据回溯选择首任务分支。

Air的TMoT在中间任务选择过residual，硬选择阶段平均0.846/0.808，均弱于对应软加权0.771/0.761。最终虽回到append或none，仍未超过旧最佳单专家0.963/0.824及summary warm ER10约0.807。说明选择信号也存在时序错配，不能仅把问题归因于软融合。

这是机制诊断，不作为最终TMoT方案。若继续研究可靠性，需要同时处理无历史证据的初始化和训练前/训练后误差排序变化，保持只使用已到达TRAIN的时序约束；不得在VAL上按阶段选择专家。三专家训练/存储成本仍保留，硬选择不降低维护成本。

独立从TRAIN预测和身份重算累计SSE、计数与最小误差索引；独立标量选择与加权结果一致，各阶段指标和VAL身份重算。源文件SHA在执行前后保持，清单存入JSON。所有正负结果保留，分类与一般时序回归总体目标不缩减。
