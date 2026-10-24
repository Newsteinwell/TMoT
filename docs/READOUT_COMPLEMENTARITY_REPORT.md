# v65 读出错误互补性诊断

只读v64的已核验VAL预测，覆盖144个候选/参考成对任务流；不训练、不改模型、不访问TEST。两个解析例子测试通过，覆盖类别不平衡、错误分解与两者共同出错。每阶段复核样本身份、候选与参考宏准确率。

下表排除seed42/order17，平均其余五个组合。数值按类别平衡，单位为百分点。纠错=参考错误而候选正确；损害=参考正确而候选错误；净增益=纠错−损害。标签oracle只表示知道答案后在两个硬预测中选择的诊断值，不是可实现方法，也不是一般得分融合的上限。

| 数据集 | 表示 | 候选 | 参考 | 最终纠错 | 最终损害 | 净增益 | 两者均错 |
|---|---|---|---|---:|---:|---:|---:|
| handwriting | mantis | bounded | fecam | 0.000 | 4.231 | -4.231 | 65.769 |
| handwriting | mantis | bounded | ridge | 6.923 | 28.846 | -21.923 | 41.154 |
| handwriting | mantis | original | fecam | 23.462 | 12.692 | +10.769 | 42.308 |
| handwriting | mantis | original | ridge | 0.000 | 6.923 | -6.923 | 48.077 |
| handwriting | summary | bounded | fecam | 1.538 | 4.615 | -3.077 | 65.769 |
| handwriting | summary | bounded | ridge | 18.077 | 3.846 | +14.231 | 66.538 |
| handwriting | summary | original | fecam | 0.769 | 16.538 | -15.769 | 66.538 |
| handwriting | summary | original | ridge | 4.615 | 3.077 | +1.538 | 80.000 |
| lsst | mantis | bounded | fecam | 3.260 | 5.797 | -2.537 | 33.410 |
| lsst | mantis | bounded | ridge | 10.016 | 7.968 | +2.048 | 31.240 |
| lsst | mantis | original | fecam | 5.907 | 9.182 | -3.275 | 30.764 |
| lsst | mantis | original | ridge | 1.651 | 0.341 | +1.310 | 39.604 |
| lsst | summary | bounded | fecam | 4.315 | 7.588 | -3.273 | 51.401 |
| lsst | summary | bounded | ridge | 20.319 | 5.579 | +14.741 | 53.410 |
| lsst | summary | original | fecam | 5.724 | 20.716 | -14.991 | 49.991 |
| lsst | summary | original | ridge | 10.731 | 7.709 | +3.022 | 62.998 |
| phonemespectra | mantis | bounded | fecam | 5.732 | 5.520 | +0.211 | 58.401 |
| phonemespectra | mantis | bounded | ridge | 12.217 | 8.899 | +3.318 | 55.023 |
| phonemespectra | mantis | original | fecam | 8.296 | 10.437 | -2.142 | 55.837 |
| phonemespectra | mantis | original | ridge | 8.356 | 7.391 | +0.965 | 58.884 |
| phonemespectra | summary | bounded | fecam | 4.434 | 4.103 | +0.332 | 84.555 |
| phonemespectra | summary | bounded | ridge | 8.869 | 4.555 | +4.314 | 84.103 |
| phonemespectra | summary | original | fecam | 4.646 | 6.063 | -1.418 | 84.344 |
| phonemespectra | summary | original | ridge | 7.270 | 4.706 | +2.564 | 85.701 |

## 对下一步的约束

PhonemeSpectra+Mantis的bounded读出相对FeCAM纠错5.732pp、损害5.520pp，净收益仅0.211pp。这说明候选确实含有互补的判别信息，但目前决策不能保留纠错而抑制损害；不能据此保证存在可学习且泛化的门控。LSST两种表示同样存在纠错，但损害更大。

Handwriting+Mantis最终纠错为0，损害4.231pp；即使事后知道真实标签，在这两个硬预测间选择也不能超过FeCAM。其最终预测分歧达41.54%，但多数分歧是两者都错。因此分歧大不能当作互补信息强，也不能依靠更复杂的选择器解决该条件。

下一轮应分别验证两个问题：已有互补条件能否用不依赖评估标签的可靠性规则控制损害；缺少互补条件是否需要改变表示或学习目标。任何规则只能在预先指定的开发配置确定，随后冻结复验，不按评估标签选择专家，不报告标签oracle为算法成绩。

本诊断仅针对分类硬预测，不证明回归模块有效。一般多维时序持续学习目标保持不变，分类与回归的收益须各自验证。

## 磁盘维护

对v31/v43/v44的212个已完成旧检查点执行无损压缩检查；解压SHA验证后才删除原始副本，低收益项保留原文件。1739个非模型证据文件SHA保持不变。回收2.49GiB，剩余4.84GiB。当前v54/v55模型引用及v63/v64统计不受影响。归档清单与逐文件校验见results/old_history_archive_v65.json和journal。SSH末尾断开后重新核对报告complete=true，未重复运行。
