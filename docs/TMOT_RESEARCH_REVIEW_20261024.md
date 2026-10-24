# TMoT 阶段性实验总结：截至 v81

日期：2026-10-24。状态：本轮训练与核验结束，自动实验目标已暂停，不启动下一轮。

## 一、当前判断

**研究目标仍然是面向一般多维时间序列持续学习的算法；当前证据还没有证明我们已经实现这一目标。** 现在具备分类、类别增长和回归的可运行研究框架，若干改进在特定条件下有效，但没有一个固定版本在多数数据集上稳健超过强基线。不能把“程序支持多个任务类型”写成“算法在这些任务上普遍有优势”。

当前最值得保留的三项结果是：

1. 原始 WiMoT 迁移实现了历史统计与任务适配，但完整迁移在早期分类结果中没有超过 SLDA。
2. 任务级合成回放 task256 明显强于严格小样本、冷启动的 ER10；然而它使用更多统计/合成存储，也未普遍超过 FeCAM、ridge 和 warm ER10。
3. 类几何与相对间隔选择器在 PhonemeSpectra + Mantis 上有多种子/顺序的局部收益：相对 FeCAM 最终平均 +1.026 个百分点、阶段平均 +1.340 个百分点。其他难分类数据集没有同样收益，且遗忘指标变差。

v81 是最近完成的实验，不是当前最优统一方法。版本号包含诊断、基线补充和失败试验，不表示 81 次连续性能升级，也不等于 81 个独立算法。

## 二、我们实际研究了哪些方法

| 方法族 | 核心做法 | 当前定位 |
|---|---|---|
| 原始 WiMoT 迁移 | OSCL、白化、谱解码、均值先验、Student-t 校准、历史矩与仿射传输；类别新增适配、回归另设连续标签读出 | 忠实迁移与初始参照，性能未确立优势 |
| 统计/合成历史回放 | 高斯或局部混合历史分布；匹配特征与目标矩；精确保留线性统计；训练非线性表示 | 有效解决部分实现和数值问题，不能保证非线性历史风险被正确重建 |
| task256 任务级合成 | 为历史任务保留合成代表，训练特征与读出；对固定统计路径保留精确基座 | 当前主要可比较候选；优于弱小缓存 ER，但非总体赢家 |
| 类条件几何与选择器 | 借鉴 FeCAM 的类均值、协方差和距离几何；有界统计补全；按相对分类间隔选择分支 | PhonemeSpectra 局部最有希望，未跨数据集成立 |
| 时序先验与残差 | 使用可观测的历史目标末值，预测残差；尝试只用到达训练数据选择先验 | Beijing 大幅改善主要来自共享先验，算法额外收益小 |
| 蒸馏、支持点及残差修正 | 历史统计教师、合成支持点、温度/硬标签、非线性残差网络 | 多个版本比自己的旧版好，但仍不及强参照 |
| v81 交互表示 | 增加通道自身及跨通道滞后相关特征，再使用任务合成/ridge/FeCAM | 部分表示获益，基线同样受益，无普遍优势 |

后期分支并非都保留原始 WiMoT 的全部模块；不能把所有结果归到一个从头至尾不变的“完整 TMoT”上。

## 三、最关键的数值证据

### 3.1 原始迁移：没有超过简单强基线

早期已有 TEST 结果如下，单位为准确率%，均值±标准差；这些 TEST 在早期探索中已经被查看，不能再作为最终独立盲测。

| 数据集 | 完整迁移 TMoT | 同表示 SLDA | 同表示 ridge |
|---|---:|---:|---:|
| HAR | 96.14±0.26 | 96.35±0.35 | 96.25±0.35 |
| UWave | 68.36±5.95 | 73.67±5.33 | 67.66±4.63 |
| DailySports | 92.08±1.02 | 92.54±1.07 | 91.90±1.30 |

历史项在部分消融中有帮助，UWave 的语义补全也改善过自身版本，但不足以证明完整方法总体优于 SLDA。依据：[第二轮报告](ROUND2_REPORT_2026-10-24.md)。

### 3.2 主候选 task256：对比对象决定结论

下表是难分类集固定 seed42/order17 的最终 VAL 宏准确率，不是多种子确认。Mantis 与 summary 是两种表示，同一个数据集不能重复计为两个数据集。

| 数据集/表示 | task256 | 原表示的较强统计基线 | 差值（百分点） |
|---|---:|---:|---:|
| Handwriting / Mantis | 44.231 | ridge 51.923 | -7.692 |
| Handwriting / summary | 19.231 | FeCAM 32.692 | -13.461 |
| LSST / Mantis | 59.607 | FeCAM 62.063 | -2.456 |
| LSST / summary | 31.701 | FeCAM 38.837 | -7.136 |
| PhonemeSpectra / Mantis | 34.842 | FeCAM 35.596 | -0.754 |
| PhonemeSpectra / summary | 9.955 | FeCAM 11.463 | -1.508 |

v57 的严格缓存实验中，task256 在 12/12 个“数据集×表示”条件下最终分数超过普通冷启动 ER10，11/12 超过 moment ER10；但其中四项阶段平均较差。加入 warm ER10 后，回归部分优势被反转。例如 Air/summary：task256 为 0.8265，warm ER10 为 0.8073（RMSE 越低越好）。

因此“任务合成比严格小缓存冷启动 ER 更强”有证据；“任务合成比目前强基线普遍更强”没有证据。依据：[缓存对照](TASK_BUDGET_REPORT.md)、[FeCAM](FECAM_BASELINE_REPORT.md)、[统计读出](ANALYTIC_HEADS_REPORT.md)。

### 3.3 最好的局部改进：v67 相对间隔选择器

固定规则，在剔除原 seed42/order17 控制后五个种子/顺序组合上，相对 FeCAM 的平均差值：

| 数据集/表示 | 最终准确率差（pp） | 阶段平均差（pp） | 最终胜出次数 |
|---|---:|---:|---:|
| Handwriting / Mantis | -2.308 | -2.787 | 0/5 |
| Handwriting / summary | -1.538 | -3.488 | 0/5 |
| LSST / Mantis | -0.519 | -1.969 | 2/5 |
| LSST / summary | -0.657 | -1.516 | 1/5 |
| PhonemeSpectra / Mantis | **+1.026** | **+1.340** | **5/5** |
| PhonemeSpectra / summary | +0.905 | +0.183 | 5/5 |

PhonemeSpectra/Mantis 是现阶段最清晰的局部正结果，但旧任务遗忘平均增加 0.806pp。summary 的阶段平均只有 2/5 为正，且遗忘增加 1.002pp。五个组合仍使用反复开发过的 VAL，不是新盲测。Handwriting/Mantis 若与更强的 ridge 比，差距更大。

依据：[v67 完整轨迹与遗忘报告](MARGIN_STREAM_REPORT.md)。不能将此结果外推为分类全面有效，更不能外推为回归有效。

### 3.4 回归：大幅提升主要来自共享时序先验

v68 固定开发配置的结果：

| 数据集/表示与形式 | TMoT 最终 RMSE | 关键基线 | 解读 |
|---|---:|---:|---|
| Air / Mantis，无先验 | 0.962599 | warm ER10 0.965837 | 约0.34%改善，未到5% |
| Air / summary，无先验 | 0.826536 | warm ER10 0.807300 | TMoT更差 |
| Beijing / Mantis，残差形式 | 27.341932 | 同形式ridge 27.695442 | 约1.28%改善，未到5% |
| Beijing / summary，残差形式 | 27.092263 | 同形式ridge 27.126868 | 约0.13%改善，且阶段平均更差 |

Beijing 从约44下降到约27看起来很大，但同样的历史末值/残差形式也把 ridge 降到了约27。不能把这一整段下降算成 TMoT 的贡献。Air 的残差形式反而恶化。v69–72 的自适应先验尝试没有形成统一强方案。

真实回归验证目前主要是两个标量预测数据集；多输出支持的单元/合成测试不等于真实多输出回归上的效果验证。依据：[因果参考回归对照](CAUSAL_TASK_REPORT.md)。

### 3.5 最新 v81：有局部表示收益，没有改变总判断

20组共享训练、312条轨迹、3288个阶段；全部最终预测由保存状态精确复现。单种子、固定顺序，各方法按最终 VAL 选择超参数。

| 分类数据/表示 | 原task256 | 加交互后的task256 | 加交互后的较强统计基线 |
|---|---:|---:|---:|
| Handwriting / Mantis | 44.231 | 51.923 | ridge 51.923 |
| Handwriting / summary | 19.231 | 26.923 | FeCAM 32.692 |
| LSST / Mantis | 59.607 | 59.120 | FeCAM 60.467 |
| LSST / summary | 31.701 | 36.033 | FeCAM 41.711 |
| PhonemeSpectra / Mantis | 34.842 | 34.087 | FeCAM 35.747 |
| PhonemeSpectra / summary | 9.955 | 14.027 | FeCAM 13.273 |

Handwriting/Mantis 最终追平 ridge，但阶段平均为46.189%，低于ridge49.023%。Handwriting/summary 最终提高，阶段平均却由21.086%降至13.346%，说明只看最终分数会误判。

PhonemeSpectra/summary 相对同输入 FeCAM 提高0.754pp，未达到1pp门槛；Mantis退步。LSST仍低于FeCAM，而且Mantis新表示下的FeCAM也低于原表示FeCAM。

回归中，Air/summary 从0.826536降到0.821686，仍不如warm ER10的0.807300；Air/Mantis最终变差。Beijing/Mantis为27.576818，对照ridge27.258569；summary为27.039754，对照ridge26.926455，两项均未超过同输入ridge。新增64维也增加了容量与统计存储，并非等维/等内存试验。

完整结果与全部超参数：[v81报告](INTERACTION_REPORT.md)。

## 四、各轮改进究竟留下了什么

下表按机制归并所有主要实验分支。诊断性上界、全历史离线重算、基线扩展不能计成可部署方法的胜场。

| 轮次 | 改进/验证 | 实际结果与保留价值 | 证据 |
|---|---|---|---|
| 初始–v6 | 原始迁移、维数、历史损失、语义补全 | 局部消融有效，完整迁移未超过SLDA | [报告](ROUND2_REPORT_2026-10-24.md) |
| v7–9 | 核特征与统计传输 | 无总体优势；部分变化来自正则化而非新增信息 | [报告](ROUND3_VALIDATION_REPORT.md) |
| v10–12 | 高斯合成与非线性残差 | 常弱于同训练预算ER，简单集存在天花板 | [报告](NONLINEAR_DEVELOPMENT_REPORT.md) |
| v14 | 局部混合矩 | 缓解单高斯近似，但非稳定胜出 | [报告](LOCAL_MOMENT_REPORT.md) |
| v15–16 | 目标联合矩、有限批矩校准 | 将矩误差压到约1e-8或更小；预测收益不稳定 | [目标矩](TARGET_MOMENT_REPORT.md)、[有限回放](FINITE_REPLAY_REPORT.md) |
| v17–18 | 直接局部专家、共享ridge先验 | 没有总体优势；已知域信息也未自动解决问题 | [专家](DIRECT_EXPERT_REPORT.md)、[共享先验](SHARED_PRIOR_EXPERT_REPORT.md) |
| v19 | 因果末值与残差 | Beijing明显改善，但简单ridge也获益 | [报告](CAUSAL_REFERENCE_REPORT.md) |
| v20–21 | 五分类集多种子/顺序 | 没有总体优势；部分数据达100%，区分度不足 | [报告](CLASSIFICATION_ROBUSTNESS_REPORT.md) |
| v22–23 | 维度、投影种子 | 某些数据提高、另一些下降，无统一最佳宽度 | [维度](PROJECTION_DIAGNOSTIC_REPORT.md)、[种子](PROJECTION_SEED_REPORT.md) |
| v24–25 | 证据优化、ARD读出 | 优化或收敛改进未转化成普遍VAL收益 | [读出](EVIDENCE_READOUT_REPORT.md)、[精化](EVIDENCE_REFINEMENT_REPORT.md) |
| v26–30 | CE/Huber、联合时序、末值、尺度传输、邻居 | 局部改善往往基线也有；无统一胜出 | [损失](TASK_LOSS_REPORT.md)、[联合时序](TEMPORAL_JOINT_REPORT.md)、[邻居](NEIGHBOR_REPORT.md) |
| v31–34 | 暖启动、局部细分、支持阈值、合并 | 部分改善；过细分区曾包含单样本，不能称无样本泄露 | [暖启动](WARM_TRANSPORT_REPORT.md)、[支持](LOCAL_SUPPORT_REPORT.md)、[合并](LOCAL_MERGE_REPORT.md) |
| v35 | 历史梯度诊断 | 大量高斯样本仍有梯度偏差，二阶矩不足以确定非线性风险 | [报告](GRADIENT_DIAGNOSTIC_REPORT.md) |
| v36–39 | 固定函数、历史函数、精确线性基座 | 精确保留线性统计是有效工程改进，仍未超强基线 | [函数](FUNCTIONAL_REPORT.md)、[精确基座](EXACT_BASE_REPORT.md) |
| v40–41 | 正交化、正则化创新 | 裸伪逆出现严重数值失败；正则化修复稳定性，无普遍性能优势 | [正交](ORTHOGONAL_REPORT.md)、[正则化](REGULARIZED_INNOVATION_REPORT.md) |
| v42–44 | 历史权重、稳健性、累计归一化 | DailySports局部+0.567pp，HAR略差、UWave持平；不能推广 | [稳健性](HISTORY_ROBUSTNESS_REPORT.md)、[归一化](ARRIVED_CALIBRATION_REPORT.md) |
| v45–48 | 难数据、完整特征、潜变量、全协方差 | 发现表示/压缩瓶颈，但完整历史诊断不等于在线算法 | [难数据](HARDER_VALIDATION_REPORT.md)、[完整特征](FULL_FEATURE_REPORT.md)、[协方差](COVARIANCE_REPLAY_REPORT.md) |
| v49–51 | 真历史矩、联合读出、经验回放 | 真矩或更低TRAIN损失也不保证更高VAL；经验代表更接近部分目标矩 | [精确读出](EXACT_READOUT_DIAGNOSTIC_REPORT.md)、[经验回放](EMPIRICAL_REPLAY_REPORT.md) |
| v52–54 | 有限合成、任务感知、容量扩展 | 形成task256候选；匹配目标下降不等于泛化提高 | [任务合成](TASK_SYNTHETIC_REPORT.md)、[容量](SYNTHETIC_CAPACITY_REPORT.md) |
| v55 | task256多种子/顺序 | 存在局部增益与明显负结果，不是统一优势；旧ER非10%约束 | [报告](SYNTHETIC_ROBUSTNESS_REPORT.md) |
| v56 | 合成漂移方案 | 找到协议，未找到可确认的完成结果；不计入已完成证据 | [协议](SYNTHETIC_DRIFT_PROTOCOL.md) |
| v57–58 | 严格ER10、warm ER10 | task256胜冷启动小缓存；warm版本揭示回归优势不足 | [预算结果](TASK_BUDGET_REPORT.md)、[warm协议](WARM_BUDGET_PROTOCOL.md) |
| v59–60 | FeCAM、NCM、SLDA、ridge | 强统计基线暴露原先对照不足，不能只看ER | [FeCAM](FECAM_BASELINE_REPORT.md)、[统计读出](ANALYTIC_HEADS_REPORT.md) |
| v61–64 | 类距离、联合补全、有界化、复验 | 有界化消除统计约束违规；单次Phoneme约+1.207pp，多组合均值仅+0.211pp | [有界](BOUNDED_COMPLETION_REPORT.md)、[复验](BOUNDED_ROBUSTNESS_REPORT.md) |
| v65–67 | 互补性、相对间隔选择、完整流复验 | Phoneme/Mantis局部稳健正收益，其余难集失败 | [完整流](MARGIN_STREAM_REPORT.md) |
| v68–72 | 回归先验、首任务校准、在线选择 | 恢复Beijing强参照，未取得5%算法额外优势；Air适配不足 | [因果任务](CAUSAL_TASK_REPORT.md)、[首任务选择](FIRST_CALIBRATION_REPORT.md) |
| v73–74 | FT、LwF式蒸馏、统计教师 | 仅当前输入蒸馏导致严重旧类遗忘；强教师不等于强学生 | [顺序头](SEQUENTIAL_HEAD_REPORT.md)、[教师](STATISTICAL_TEACHER_REPORT.md) |
| v75–78 | 历史支持点、置信度、锐化/硬标签、互补性 | 大幅改善弱蒸馏版本，最终仍落后强统计基线 | [支持](SUPPORT_TEACHER_REPORT.md)、[置信度](TARGET_CONFIDENCE_REPORT.md)、[互补性](TEACHER_COMPLEMENTARITY_REPORT.md) |
| v79–80 | 暖/冷非线性残差修正 | 训练拟合改善，VAL仍不及旧强基线；冷启动不统一更好 | [warm](CAUSAL_RESIDUAL_REPORT.md)、[cold](COLD_RESIDUAL_REPORT.md) |
| v81 | 跨通道滞后交互 | 局部表示收益，尚未转化为方法相对强基线的多数优势 | [报告](INTERACTION_REPORT.md) |

v79/v80 尤其说明“训练成功”与“泛化成功”不同：v79局部合成24/24配置的历史TRAIN拟合改善，但仅7/24改善相对基座的VAL。v80改变初始化后仍未成为更好的统一方案。没有依据继续把这些分支逐项叠加并假定效果会累积。

## 五、当前比较的公平性与证据强度

**回放预算。** 用户确定的口径是每个历史任务最多保存该任务训练样本的10%，总缓存随历史任务数增长。v57起严格执行 floor(N_task/10)。早期4MiB/64MiB缓存经常装下全部历史，不能与后来的ER10混称；保留为历史/大缓存参照。

**总内存并不相等。** 限制真实样本数，不等于限制统计矩阵、合成代表和模型的字节数。例如Phoneme/Mantis的task256约158.67MiB，普通ER10约4.34MiB；Handwriting约69.81对3.72MiB；Air约12.28对3.48MiB。因此不能宣称当前优势是在相同总存储预算下获得，也不能把“无真实回放”自动写成隐私保证。

**基线并非只有ER。** 已有ridge、SLDA、NCM、适配FeCAM、普通/矩匹配/暖启动ER10、FT及LwF式蒸馏等结果。覆盖不完全，每个版本并非都跑过所有基线。早期RanPAC式实现也不等于完整官方复现。目前没有可作为完整结果引用的EWC、DER++、A-GEM实验。FeCAM等迁移实现应描述为时序适配版，而非冒称原论文官方复现。

**开发集反复选择。** 多轮使用同一VAL，包括挑超参数、分支与表示；有些复验增加了种子/顺序，但不是独立新数据。v81每格也是最终VAL最优配置，不能组成一个无需选择的统一算法。新轮次没有读取TEST，不代表早期TEST从未被看过。

**验证范围。** 类别新增已能运行；完整域增长与真实多输出回归证据较少。两种表示不能算两个数据集；同一训练的多个读出参数不能算独立重复。模型重载一致、矩误差很小、无标签泄漏等核验是可信实验的必要条件，并不是性能优势证据。

## 六、距离原定目标还有多远

既定验收集合为8个分类集与2个回归集；要求至少6/10有明确改善，其中至少5/8分类、1/2回归；分类至少+1pp、回归相对RMSE改善至少5%，同时阶段平均不能实质恶化，并计划以多种子/任务顺序检查稳健性。参见[难数据协议](HARDER_DATASETS_PROTOCOL.md)。

**该标准尚未达到。** 不能把不同版本在各自最有利数据集上的最好点拼成一个统一方法的总体结果。目前也没有一张“同一固定最终方法、完整验收集合、规定重复、强基线”的完成表。Phoneme的局部结果值得保留，但不足以替代这个缺口。

这不证明一般多维时序持续学习的研究方向不可行；它说明当前候选机制和实验证据还不足以支撑我们希望提出的性能主张。若现在写论文，可以准确报告框架、适配、诊断和局部收益，不能据此声称已经取得多数数据集稳健领先，或已经具备顶会竞争力的完整实证。

## 七、停止状态与材料保存

本轮完成后已停止推进，自动实验目标处于暂停状态；现场进程检查未发现训练或v81核验仍在运行。服务器数据盘约15GiB可用，本次没有再启动训练或删除材料。

v81训练源版本为6bd3d4575dc6fcb4eaab798ad8db43ef15b1b0ff。完整审计保存于`results/interaction_v81_audit.json`，选中配置保存于`results/interaction_v81_selected.json`，包含负结果的完整参数表见v81报告。

历次清理已删除部分旧探索检查点，因此不能声称所有旧模型均可直接重载；保留的配置、预测、指标与审计材料用于回溯。具体清理清单见[磁盘清理记录](DISK_CLEANUP_20261024.md)。本总结不以失败模型被清理为理由抹除负结果，也不把新版本替换旧结果。
