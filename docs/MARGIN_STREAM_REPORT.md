# v67 相对间隔选择的完整任务流复验

训练源码8a38d119ae7738b9afffd28b0c8a8ac6df474128。36组共享训练、216条完整轨迹。规则和所有头参数固定，无新增阈值；原五个头逐阶段复现v64，最终距离和选择精确复现v66。端到端Handwriting烟测通过，规则的三项测试见v66。

每行平均除seed42/order17控制外的五个组合；控制和所有单次结果保存在JSON。重复使用开发TRAIN/VAL，不是独立盲测；无TEST。种子/顺序组合不是独立数据集。单位为宏准确率百分点差。

| 数据集 | 表示 | 参考 | 最终均值差 | 阶段平均差 | 阶段平均差范围 | 最终胜出/5 | 阶段平均胜出/5 |
|---|---|---|---:|---:|---:|---:|---:|
| handwriting | mantis | bounded | +1.923 | +1.834 | [-1.158, +4.521] | 3 | 3 |
| handwriting | mantis | fecam | -2.308 | -2.787 | [-9.394, +3.475] | 0 | 2 |
| handwriting | mantis | original | -13.077 | -15.541 | [-18.711, -12.310] | 0 | 0 |
| handwriting | mantis | ridge | -20.000 | -16.338 | [-21.203, -12.689] | 0 | 0 |
| handwriting | summary | bounded | +1.538 | +1.912 | [-0.565, +4.520] | 3 | 4 |
| handwriting | summary | fecam | -1.538 | -3.488 | [-6.239, -1.677] | 0 | 0 |
| handwriting | summary | original | +14.231 | +8.090 | [+0.755, +13.763] | 5 | 5 |
| handwriting | summary | ridge | +15.769 | +8.276 | [+0.762, +14.576] | 5 | 5 |
| lsst | mantis | bounded | +2.018 | +2.638 | [+1.001, +4.069] | 5 | 5 |
| lsst | mantis | fecam | -0.519 | -1.969 | [-3.068, -1.183] | 2 | 0 |
| lsst | mantis | original | +2.756 | +0.497 | [-2.982, +2.832] | 5 | 3 |
| lsst | mantis | ridge | +4.066 | +1.012 | [-0.109, +1.831] | 5 | 4 |
| lsst | summary | bounded | +2.616 | +2.127 | [-0.809, +4.355] | 5 | 3 |
| lsst | summary | fecam | -0.657 | -1.516 | [-3.222, +0.498] | 1 | 2 |
| lsst | summary | original | +14.334 | +12.167 | [+5.973, +16.542] | 5 | 5 |
| lsst | summary | ridge | +17.356 | +14.499 | [+13.838, +15.475] | 5 | 5 |
| phonemespectra | mantis | bounded | +0.814 | +0.401 | [+0.034, +0.720] | 4 | 5 |
| phonemespectra | mantis | fecam | +1.026 | +1.340 | [+0.997, +1.879] | 5 | 5 |
| phonemespectra | mantis | original | +3.167 | +1.921 | [+0.928, +2.651] | 5 | 5 |
| phonemespectra | mantis | ridge | +4.133 | +4.539 | [+3.533, +5.129] | 5 | 5 |
| phonemespectra | summary | bounded | +0.573 | +0.639 | [+0.331, +1.052] | 4 | 5 |
| phonemespectra | summary | fecam | +0.905 | +0.183 | [-0.498, +1.356] | 5 | 2 |
| phonemespectra | summary | original | +2.323 | +2.208 | [+1.075, +3.127] | 5 | 5 |
| phonemespectra | summary | ridge | +4.887 | +4.787 | [+2.801, +6.192] | 5 | 5 |

## 旧任务遗忘与后向迁移

遗忘：各旧任务从学到该任务起至最终阶段的最高准确率减最终准确率，含最终阶段所以非负；BWT：最终准确率减该任务刚学完时准确率。对旧任务等权平均，排除最后任务。以下为相对参考的遗忘减少量与BWT增量，正值更好。

| 数据集 | 表示 | 参考 | 遗忘减少（pp） | BWT增量（pp） |
|---|---|---|---:|---:|
| handwriting | mantis | bounded | +2.500 | -0.833 |
| handwriting | mantis | fecam | -4.167 | -2.083 |
| handwriting | mantis | original | +4.583 | -2.500 |
| handwriting | mantis | ridge | -5.417 | -10.000 |
| handwriting | summary | bounded | -0.833 | +1.667 |
| handwriting | summary | fecam | +1.667 | +4.583 |
| handwriting | summary | original | +12.917 | +9.583 |
| handwriting | summary | ridge | +17.917 | +7.917 |
| lsst | mantis | bounded | +0.789 | +0.810 |
| lsst | mantis | fecam | +0.478 | +1.156 |
| lsst | mantis | original | +0.446 | -0.009 |
| lsst | mantis | ridge | +0.492 | +0.504 |
| lsst | summary | bounded | -0.258 | +2.261 |
| lsst | summary | fecam | -1.398 | +1.558 |
| lsst | summary | original | -0.897 | +0.893 |
| lsst | summary | ridge | -4.924 | -5.134 |
| phonemespectra | mantis | bounded | +0.763 | -0.022 |
| phonemespectra | mantis | fecam | -0.806 | +0.501 |
| phonemespectra | mantis | original | +3.322 | +2.843 |
| phonemespectra | mantis | ridge | -0.763 | -0.664 |
| phonemespectra | summary | bounded | +0.109 | -0.458 |
| phonemespectra | summary | fecam | -1.002 | +2.157 |
| phonemespectra | summary | original | +2.135 | +2.898 |
| phonemespectra | summary | ridge | -2.593 | +0.338 |

## 核验与范围

每阶段从保存的全部类别距离独立排序重算选择标志及预测；核对两分支原预测、输入ID、标签、各任务与总体宏准确率。最终选择和距离矩阵复现v66；所有原五头预测复现v64。冻结参数、输入哈希与复用检查点SHA逐项核对。训练结束时网络、g/k、合成点和RNG精确复现v55，单位化统计精确复现v64。

选择函数不访问评估标签。训练只使用当前任务真实样本、历史累计统计和合成点；保存的VAL距离仅用于核验。引用旧模型避免重复磁盘副本，并不代表算法没有内存成本。

总体目标仍为一般多维时序持续学习。本轮只检验类增长条件，不将分类间隔规则直接用于连续标签或宣称回归有效。

## 结论与后续方向

PhonemeSpectra+Mantis相对FeCAM最终平均+1.026pp、阶段平均+1.340pp，五个非原控制组合两项均为正。但旧任务遗忘平均增加0.806pp，不能声称该规则一致改善抗遗忘。遗忘比较各方法自己的历史最高值，BWT比较各自刚学完时的值；两者基准不同，因此BWT改善与遗忘增加可以同时发生。

PhonemeSpectra+summary最终平均+0.905pp且5/5为正，阶段平均仅+0.183pp、2/5为正，旧任务遗忘增加1.002pp。两个Handwriting和两个LSST条件相对FeCAM的最终与阶段平均差均为负。不能接受为多数数据集稳健优势，不把这一选择器升级为通用最终方案。

分类下一步需要区分置信代理不可比与学习分支本身缺乏互补信息；不能只提高选择阈值后在同一批VAL上反复挑选。回归下一步需重新对齐已知强时间先验：Beijing旧开发持久性残差ridge约27.13 RMSE，而近期通用读出约44，须保留并复核这个强参考。该先验仅适用于历史目标可观测的预测场景，不能把它当作任意连续标签回归的通用结构。总体目标不缩减为Phoneme分类或特定预测场景。
