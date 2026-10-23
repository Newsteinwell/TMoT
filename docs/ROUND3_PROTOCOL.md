# 第三轮：多数数据集明确增益与整体稳健性

## 用户确认的目标
用户选择“多数数据集有明确增益，整体优势稳健”，并继续要求一般多维时间序列持续学习主题，包含类别增长和回归。本轮不以单数据集最好数字或GPU使用率作为成功标准。

## 固定验收口径（在新测试集读取前）
- 主要新验证集合：NATOPS、Epilepsy、RacketSports、PenDigits、ArticularyWordRecognition（分类），Beijing PM2.5、Air Quality（回归）。下载、维度和样本核验尚在进行，任何不可用数据的替换必须在读取结果前记录原因。
- 原HAR/UWave/DailySports/Energy保留为开发与回归检查集，不能被称为新的盲测证据。
- “明确增益”工作阈值：相对同数据集最强的预先声明强基线，分类最终宏准确率至少+1个百分点，回归最终RMSE至少降低5%；平均阶段指标也不得反向显著恶化。
- 多数：7个新数据集至少4个达到阈值，其中分类至少3个、回归至少1个；两个任务族的整体方向均为改善。若实际来源核验导致集合变化，在任何测试前重新锁定分母。
- 稳健性：每个分类集5种子×3类序；回归5种子固定时间顺序。报告均值、运行间离散度、逐序配对差值、数据集层面效果。共享测试样本的运行不当作独立数据集；不以简单运行级t检验充当跨数据泛化证据。
- 此标准是可检验的研发目标，不预先保证达成。任何失败场景保留，不删去弱数据来改变“多数”的分母。

## 数据来源与新划分
官方UEA数据页：
- https://www.timeseriesclassification.com/description.php?Dataset=NATOPS
- https://www.timeseriesclassification.com/description.php?Dataset=Epilepsy
- https://www.timeseriesclassification.com/description.php?Dataset=RacketSports
- https://www.timeseriesclassification.com/description.php?Dataset=PenDigits
- https://www.timeseriesclassification.com/description.php?Dataset=ArticularyWordRecognition （25类、9通道，官方页面已核验；实际归档形状待核验）
官方UCI：
- https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data
- https://archive.ics.uci.edu/dataset/360/air+quality

分类保留官方test，train内分层固定验证。缺少参与者ID的数据不声称受试者独立。PenDigits是重采样笔轨迹，原始时间间隔不均匀；Epilepsy包含健康参与者模拟活动，不能描述为临床诊断验证。
回归先按原始时间切域和train/val/test，再构造不交叠输入/目标窗口；缺失输入只允许因果填充并标记，缺失目标不插值伪造。Air Quality的-200、Beijing的NA需显式处理。禁止未来时间参与归一化/插补。

## 方法改进候选（先在开发val淘汰，不看新test）
1. 稳定语义残差：保留不受首阶段类别监督截断的固定语义方向，避免只有初始类别子空间。
2. 保留强统计读出：在稳定分支基础上增加可学习残差，而非用低维适配完全替代强基底；回归和分类分别使用相容损失。
3. 用历史输入-目标矩约束残差；比较零残差、无历史项、无稳定分支，验证收益来源。
4. 自适应正则化只在训练内部或统一val预算选择；所有强基线享有同等预算。方法不能通过更大的网格或未来标签获得优势。

## 基线与公平性
保留同表示累计ridge、SLDA（分类）、ER及原TMoT迁移版；回归加训练均值/持久性预测。记录基底训练、记忆字节、训练时间、推理时间；目前仍未实现统一固定字节预算，报告必须保留该限制，后续需要配套字节预算实验。
冻结Mantis是否预训练见过这些数据无法从当前信息确认；任何数据预训练重叠需明确披露，并加入非预训练表示对照，不能称新数据test为对编码器完全未见。

## 当前状态
第二轮240轨迹及15项测试通过，全部指标/输入/ID审计完成，但性能不满足新目标。第三轮处于数据核验与方法实现阶段。服务器调度上限16；磁盘剩余约104GB，允许按已授权规则清理无价值中间产物。
