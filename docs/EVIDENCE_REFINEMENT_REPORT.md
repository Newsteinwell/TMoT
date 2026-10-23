# 同一证据目标的优化收敛复核

18条VAL轨迹全部完成，无运行失败。45项测试通过；逐阶段Adam起点与v24一致，最终后验及投影梯度由CPU独立重建核验。没有使用VAL进行优化或停止选择。

## 收敛与效果分别判断

88个阶段中68个达到归一化投影梯度≤1e-6，优化器返回success的为70个，两者未混称。isotropic的44阶段全部达标；ARD仅24/44达标，部分达到300次迭代上限。不能把本轮所有ARD阶段称为已收敛，也不作全局最优保证。

| 运行 | v24最终VAL | v25最终VAL | 达标阶段/总阶段 |
|---|---:|---:|---:|
| air_quality_mantis_ard_o17_s42 | 1.8877 | 1.8445 | 4/5 |
| air_quality_mantis_isotropic_o17_s42 | 1.0109 | 1.0629 | 5/5 |
| air_quality_summary_ard_o17_s42 | 1.1273 | 1.2133 | 4/5 |
| air_quality_summary_isotropic_o17_s42 | 0.9148 | 0.9016 | 5/5 |
| articulary_mantis_ard_o17_s42 | 100.000% | 96.000% | 7/12 |
| articulary_mantis_isotropic_o17_s42 | 100.000% | 100.000% | 12/12 |
| beijing_pm25_mantis_ard_o17_s42 | 78.0931 | 87.8972 | 1/5 |
| beijing_pm25_mantis_isotropic_o17_s42 | 43.5224 | 43.5216 | 5/5 |
| beijing_pm25_summary_ard_o17_s42 | 43.9085 | 43.9302 | 1/5 |
| beijing_pm25_summary_isotropic_o17_s42 | 43.6749 | 43.6993 | 5/5 |
| epilepsy_mantis_ard_o17_s42 | 95.833% | 100.000% | 2/2 |
| epilepsy_mantis_isotropic_o17_s42 | 95.833% | 95.833% | 2/2 |
| natops_mantis_ard_o17_s42 | 94.444% | 80.556% | 2/3 |
| natops_mantis_isotropic_o17_s42 | 94.444% | 91.667% | 3/3 |
| pendigits_mantis_ard_o17_s42 | 99.529% | 99.529% | 1/5 |
| pendigits_mantis_isotropic_o17_s42 | 99.529% | 99.529% | 5/5 |
| racketsports_mantis_ard_o17_s42 | 86.607% | 83.929% | 2/2 |
| racketsports_mantis_isotropic_o17_s42 | 86.607% | 89.732% | 2/2 |

共享精度版本充分降低梯度后仍未超过既有强参考；因此它的弱势不能全部归因于60步Adam不足。ARD的最终NATOPS已达到梯度阈值却从94.444%降至80.556%，Articulary最终也达标但从100%降至96%，说明训练证据与验证泛化存在明显不一致。不能将此直接推广为全部贝叶斯方法无效；当前先验、冻结表示和高斯one-hot替代目标仍是具体设定。

本候选不推进正式TEST，也不通过继续增加优化步数期待普遍优势。保留未收敛阶段与全部负结果。下一步应改变有明确依据的表示/损失或历史约束机制，而非继续追逐该证据目标的数值最优。一般多维时序分类与回归目标、正式七数据分母不变。
