import json
from pathlib import Path
rows=json.loads(Path('results/analytic_heads_v60_comparison.json').read_text())['rows']
lines=['# 困难分类解析读出基线 v60','', '训练/统计源码66a16112dde8abd3f93f73d1cc5f32dd5291c39f；12个共享统计作业、156条完整验证轨迹。仅原开发TRAIN/VAL，seed42/order17；未访问TEST。SLDA/ridge分别在预先固定的六值网格中按最终VAL选参，各自的raw/unit表示条件分别报告。','', '## 最终宏准确率（%）','', '| 数据集 | 表示 | TMoT | FeCAM | NCM raw/unit | SLDA raw/unit | ridge raw/unit |','|---|---|---:|---:|---:|---:|---:|']
for r in rows:
 heads=r['heads'];pair=lambda m:' / '.join(f"{100*heads[n+'_'+m]['final']:.3f}" for n in ('raw','unit'))
 lines.append(f"| {r['dataset']} | {r['encoder']} | {100*r['tmot_final']:.3f} | {100*r['fecam_final']:.3f} | {pair('ncm')} | {pair('slda')} | {pair('ridge')} |")
lines+=['','## 阶段平均宏准确率（%）','', '| 数据集 | 表示 | TMoT | FeCAM | NCM raw/unit | SLDA raw/unit | ridge raw/unit |','|---|---|---:|---:|---:|---:|---:|']
for r in rows:
 heads=r['heads'];pair=lambda m:' / '.join(f"{100*heads[n+'_'+m]['average']:.3f}" for n in ('raw','unit'))
 lines.append(f"| {r['dataset']} | {r['encoder']} | {100*r['tmot_average']:.3f} | {100*r['fecam_average']:.3f} | {pair('ncm')} | {pair('slda')} | {pair('ridge')} |")
lines+=['','## 诊断结论','', '在当前六个困难数据集×表示组合中，TMoT没有获得最高最终准确率：五个由FeCAM领先，Handwriting+Mantis由原始表示累计ridge领先。TMoT均优于最近类均值，但优势不能外推到更强读出。','', '单位化对SLDA多数条件有益，对ridge多数条件反而有损。FeCAM在同样单位化条件下多数优于共享协方差SLDA，提示类别条件统计与读出匹配值得优先验证；这不是对单一机制的因果证明。Handwriting+Mantis简单ridge已达51.923%，而TMoT为44.231%，说明复杂函数学习/合成重放并非必需收益源。','', '下一步固定现有表示和训练协议，针对TMoT保存的统计量及代表点设计类别条件距离读出，与原始表示FeCAM、原始ridge并列；同时用独立诊断区分统计近似误差和学习表示造成的损失。不得只把最佳基线改名为TMoT或事后按数据集挑选方法。','', '总体研究仍覆盖分类与回归；回归侧已有强残差读出和warm ER10结果继续保留，当前不能声称普遍优势。']
lines+=['','## 配置与核查','', '| 数据集 | 表示 | SLDA raw/unit | ridge raw/unit |','|---|---|---|---|']
for r in rows:
 h=r['heads'];lines.append(f"| {r['dataset']} | {r['encoder']} | {h['raw_slda']['setting']} / {h['unit_slda']['setting']} | {h['raw_ridge']['setting']} / {h['unit_ridge']['setting']} |")
lines+=['','核查修正记录：PhonemeSpectra+summary+unit弱收缩(.001)精度矩阵的21/262144个元素未通过原逐元素rtol=1e-6/atol=1e-5检查；其对角尺度约5e5，整体相对矩阵误差约1.2e-11、相对单位阵方程残差约1.0e-11。改用整体矩阵误差<1e-9且逆矩阵方程残差<1e-9，并保持独立重算预测完全一致的要求；不改训练、预测或指标。精确逐配置数值保存在audit JSON。','', '累计统计独立从已到达TRAIN重算；SLDA精度矩阵与直接中心化协方差一致；ridge解符合直接加权正规方程；所有阶段指标与最终预测核对。36个raw ridge配置与v45全部阶段预测完全一致，确认新runner没有通过改变协议改变对照结果。','', '各表示条件都使用同一StableSemanticBasis、同一数据划分和任务顺序；unit是额外的逐样本L2单位化。NCM不调参，SLDA使用共享类内协方差和均匀先验，FeCAM使用类别条件相关矩阵距离；差异不能全部归因于单一组件。','', '本次共享实现保存完整类别二阶矩，其计数不是NCM/SLDA最小理论内存。三种基线均不保留真实样本；累计统计不计为ER样本，但内存成本应单列。','', '这些是分类开发诊断，不替代跨种子验证，也不覆盖回归。完整基线矩阵仍缺统一协议的正则化、蒸馏及其他重放方法。总体目标保持一般多维时序持续学习，当前不能仅凭对ER的优势认定达标。']
Path('docs/ANALYTIC_HEADS_REPORT.md').write_text('\n'.join(lines)+'\n')
for method in ('ncm','slda','ridge'):
 print(method,'unit better raw',sum(r['heads']['unit_'+method]['final']>r['heads']['raw_'+method]['final'] for r in rows),'TMoT wins raw/unit',sum(r['tmot_final']>r['heads']['raw_'+method]['final'] for r in rows),sum(r['tmot_final']>r['heads']['unit_'+method]['final'] for r in rows))
