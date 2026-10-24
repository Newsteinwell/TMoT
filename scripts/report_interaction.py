import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
a=json.loads((root/'results/interaction_v81_audit.json').read_text());assert a['complete']
rows=a['runs'];warm=json.loads((root/'results/warm_budget_v58_comparison.json').read_text())['rows']
lines=['# v81 跨通道滞后交互表示结果','', '## Material Passport','', f"- 状态：20个共享训练、{a['trajectories']}条轨迹、{a['stages']}阶段；全部最终预测从保存状态精确重现。",'- TRAIN/VAL开发结果，未读取TEST；单种子、固定任务顺序。','- 训练源版本：6bd3d4575dc6fcb4eaab798ad8db43ef15b1b0ff。','', '## 分类：最终/阶段平均宏准确率（%）','', '各格选该输入/方法下最终VAL最优超参数并报告同一参数的阶段平均。属于开发选择，不能当作固定方法独立确认。', '', '|数据|表示|输入|TMoT任务合成|ridge|FeCAM|','|---|---|---|---:|---:|---:|']
selected=[]
for d in ('handwriting','lsst','phonemespectra','air_quality','beijing_pm25'):
 for e in ('mantis','summary'):
  cl=d not in ('air_quality','beijing_pm25');sign=-1 if cl else 1
  for feature in ('base','interaction'):
   found={}
   for method in (('task','ridge','fecam') if cl else ('task','ridge')):
    r=min((s for s in rows if (s['config']['dataset'],s['config']['encoder'],s['config']['features'],s['config']['method'])==(d,e,feature,method)),key=lambda s:(sign*s['final'],s['config']['param']))
    selected.append(r);found[method]=r
   if cl:lines.append('|'+ '|'.join([d,e,feature]+[f"{100*found[m]['final']:.3f}/{100*found[m]['average']:.3f}" for m in ('task','ridge','fecam')])+'|')
lines+=['','## 回归：最终/阶段平均RMSE','', '|数据|表示|输入|TMoT任务合成|ridge|既有warm ER10最终|','|---|---|---|---:|---:|---:|']
for d in ('air_quality','beijing_pm25'):
 for e in ('mantis','summary'):
  er=next(s['warm_er10']['final'] for s in warm if (s['dataset'],s['encoder'])==(d,e))
  for feature in ('base','interaction'):
   found={m:next(s for s in selected if (s['config']['dataset'],s['config']['encoder'],s['config']['features'],s['config']['method'])==(d,e,feature,m)) for m in ('task','ridge')}
   lines.append('|'+ '|'.join([d,e,feature]+[f"{found[m]['final']:.6f}/{found[m]['average']:.6f}" for m in ('task','ridge')]+[f'{er:.6f}'])+'|')
lines+=['','## 本轮结论','', '交互表示使部分任务合成结果提高，但未形成多数数据集优势：Handwriting/Mantis最终追平ridge而阶段平均更低；LSST两种表示仍落后FeCAM；PhonemeSpectra/summary相对同输入FeCAM最终提高0.754pp，尚低于1pp阈值，Mantis则退步。Air/summary仍弱于既有warm ER10；Beijing两种表示均弱于同输入ridge。此轮不升级为默认版本。','', '表中的既有warm ER10使用原输入，没有本轮交互块；Beijing也没有本轮残差参考，仅作历史定位，不能将其与有先验模型的差距归为算法收益。','', '## 全部超参数','', '|数据/表示|输入|方法|参数|最终|平均|','|---|---|---|---:|---:|---:|']
for s in rows:
 c=s['config'];lines.append(f"|{c['dataset']}/{c['encoder']}|{c['features']}|{c['method']}|{c['param']:g}|{s['final']:.6f}|{s['average']:.6f}|")
lines+=['','## 解释边界','', '所有方法使用相同的新输入；回归Beijing保留相同因果残差参考，Air使用none。交互块包含通道自身时间相关和跨通道项，本轮不能将变化归因于某一项。额外64维增加了模型容量和统计内存，不能称为等维或等内存对照。','', '原表示对照训练时逐阶段匹配v54/v68 task、v60/v68 ridge、v59 FeCAM。复核从共享状态重建所有最终预测；所有阶段指标重算，样本ID/标签/输入哈希检查通过。新表示的统计只由首任务训练数据决定，样本内归一化不跨样本。','', '整体优势须同时超过原表示和新表示的强基线，不将不同编码器算作独立数据集，也不将超参数配置作为独立统计重复。当前开发选择仍需后续多种子验证。']
(root/'docs/INTERACTION_REPORT.md').write_text('\n'.join(lines)+'\n')
(root/'results/interaction_v81_selected.json').write_text(json.dumps(selected,indent=2))
print('\n'.join(lines[:52]))
