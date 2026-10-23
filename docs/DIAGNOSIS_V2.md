# 表示瓶颈和读出诊断：112条验证轨迹

## Material Passport
- mode: run / descriptive analysis
- status: all 112 trajectory summaries collected; prediction/checkpoint audit pending
- training_source: d8735b5; queue worker increase did not change training implementation
- selection: val only; means across 2 orders × 2 seeds (Energy 2 seeds)

|配置|HAR 宏准确率%|UWave 宏准确率%|DailySports 宏准确率%|Energy RMSE↓|
|---|---:|---:|---:|---:|
|TMoT 64维|90.52|78.12|88.31|77.77|
|TMoT 128维|90.75|81.25|88.60|79.55|
|TMoT 256维|91.12|87.50|89.28|87.88|
|256维训练head|84.59|67.71|70.79|73.40|
|256维当前MSE|91.12|87.50|89.23|87.88|
|256维无SupCon|91.13|88.54|89.47|87.88|
|累计ridge|91.22|89.58|89.52|85.49|
|256维ER|90.15|87.50|85.92|80.41|

## 证据及下一步
1. 类新增分类中增加适配器维度均改善结果，UWave提升9.38个百分点，HAR和DailySports仅0.60和0.96个百分点。低维瓶颈是一个有证据支持的因素，但不能解释全部差距。UWave验证集仅24条，变化离散，不能据此做显著性宣称。
2. 分类直接使用训练head显著更差，说明简单取消解析读出不是解决方案。后续检验OSCL/Student-t读出，而不是假设训练/推理一致就一定改善。
3. 当前MSE替换CE几乎不改变分类结果；删除SupCon也只有小幅改善。不能将风险目标不一致断言为唯一原因。
4. 回归方向相反：256维更差，训练head更好。需要检查小样本方差和岭正则化；不能把分类最佳维度直接套用回归。
5. 全维方阵适配器改变坐标与岭惩罚，不一定改变可表达的线性函数集合；维度结果不等于通用表示学习优势。
6. Energy的MSE/无SupCon变体在定义上等价，结果也完全一致，只作实现一致性检查，不能当成独立支持证据。

原型仍未超过所有强基线。已预先固定并启动52条模块迁移验证，包含双分支/PCA、OSCL、Student-t、历史约束及类新增CE的消融。泛化目标保持面向一般多维时间序列持续学习；目前证据局限于这些实际运行的场景。

完整逐次结果见 `results/diagnosis_v2/individual_runs.json`，均值和离散度见 `summary.csv`。
