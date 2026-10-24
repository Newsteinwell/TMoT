# 非ER基线补充与跨任务网络诊断：v58/v59

只使用原开发TRAIN/VAL、seed42/order17，未访问TEST。v58训练源码23170a3，v59训练源码b64b919。v58初次24个作业因监督头评估内存记录错误失败，原日志保留；修复后独立warm_budget_v58_retry目录24个作业完成，168条轨迹通过核查。120项单元测试通过，另完成完整Handwriting烟测。v59完成36条轨迹，输入hash、各阶段指标与最终检查点预测核查通过。

## 同一困难分类协议上的 FeCAM 结果

FeCAM是时序特征适配版，六值alpha网格按最终VAL选择；TMoT使用既有v54 task256结果，六值ridge网格。表中均为最终宏准确率（%），不是盲测结果。

| 数据集 | 表示 | TMoT | FeCAM | TMoT−FeCAM（百分点） | FeCAM alpha |
|---|---|---:|---:|---:|---:|
| handwriting | mantis | 44.231 | 36.538 | +7.692 | 1 |
| handwriting | summary | 19.231 | 32.692 | -13.462 | 2 |
| lsst | mantis | 59.607 | 62.063 | -2.457 | 10 |
| lsst | summary | 31.701 | 38.837 | -7.136 | 10 |
| phonemespectra | mantis | 34.842 | 35.596 | -0.754 | 10 |
| phonemespectra | summary | 9.955 | 11.463 | -1.508 | 10 |

TMoT在6个组合中有5个最终指标低于FeCAM，仅Handwriting+Mantis领先。该结果直接否定“凭ER10结果即可认定普遍优势”的推断。FeCAM无需保存真实样本，但保存类别协方差/精度矩阵，不能称为零内存基线。LSST、Phoneme的最佳alpha均位于网格上边界10，不能声称已穷尽其调参潜力。

## 跨任务持续更新网络的 ER10

分类为最终宏准确率百分数，回归为最终RMSE。ridge和监督头分别列出，不事后挑选合并成一个方法。所有真实缓存仍受每任务floor(N/10)限制。warm版本继承网络和扩展监督头，AdamW状态在每阶段重置。

| 数据集 | 表示 | TMoT | warm ER10 ridge | warm ER10+矩 ridge | warm ER10监督头 |
|---|---|---:|---:|---:|---:|
| air_quality | mantis | 0.963 | 0.966 | 0.975 | 0.978 |
| air_quality | summary | 0.827 | 0.807 | 0.886 | 0.818 |
| beijing_pm25 | mantis | 43.912 | 48.611 | 45.091 | 50.237 |
| beijing_pm25 | summary | 44.031 | 38.903 | 44.919 | 40.169 |
| energy | mantis | 64.864 | 69.692 | 66.783 | 70.011 |
| energy | summary | 66.531 | 69.146 | 67.461 | 68.476 |
| handwriting | mantis | 44.231 | 9.615 | 42.308 | 7.692 |
| handwriting | summary | 19.231 | 9.615 | 15.385 | 3.846 |
| lsst | mantis | 59.607 | 46.852 | 60.319 | 45.461 |
| lsst | summary | 31.701 | 24.525 | 24.249 | 27.246 |
| phonemespectra | mantis | 34.842 | 20.060 | 28.658 | 16.591 |
| phonemespectra | summary | 9.955 | 6.938 | 7.391 | 5.430 |

暖启动并非统一改善，但暴露出回归的新强对照：Beijing+summary warm ER10 ridge RMSE约38.90，优于TMoT的44.03；Air+summary约0.8073也优于TMoT约0.8265。旧的简单残差读出仍须保留（Beijing已有约27的历史验证结果，需单独核对配置）。阶段平均结果保存在JSON，不能仅根据此最终表判定全过程优势。

v58核查包括144个非监督头配置首阶段与v57预测完全一致、所有缓存选样索引一致、跨阶段旧网络初始化误差为0；带矩和不带矩版本监督头逐阶段预测一致，确认矩统计只作用于解析读出。

## 对后续研究的影响

保留一般多维时序持续学习目标。优先研究分类侧类别条件统计与读出匹配，以及回归侧表示/读出尺度问题；不能通过削弱基线或只展示ER结果制造优势。继续补齐统一协议的SLDA、随机特征累计ridge、顺序微调、正则化/蒸馏/其他重放基线，再执行固定配置的多种子与新任务顺序验证。当前证据不足以达到稳健优势验收条件。

已有非ER历史结果详见NON_ER_BASELINE_INVENTORY.md及对应JSON，区分历史配置和新统一协议，RanPAC风格基线不冒充完整原论文复现。

## 磁盘

FeCAM全部36个完成检查点已尝试无损gzip归档并校验解压SHA；原预测/日志等579份非模型文件hash未变。回收约1.34GiB，归档后剩余约6.79GiB。归档清单见results/fecam_archive_v59.json；需要重载检查点时使用scripts/restore_archived_checkpoint.py恢复。
