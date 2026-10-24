import json
from pathlib import Path
runs=json.loads(Path('results/sequential_head_v73_audit.json').read_text())['runs'];old=json.loads(Path('results/warm_budget_v58_comparison.json').read_text())['rows'];rows=[];lines=['# v73 顺序微调与输出蒸馏对照','','48条完整轨迹完成，四项单元测试通过；所有首阶段预测与v58 warm ER10监督头逐值一致，所有阶段指标重算、最终检查点预测复算与输入哈希核验通过。训练源码3e0344ef285f081a0ba542f47a305b0e4df408c2。仅开发TRAIN/VAL、seed42/order17，无TEST。','','分类数字为宏准确率百分数（越高越好），回归为RMSE（越低越好）。蒸馏强度按最终VAL在预先固定的0.1/1/10中选择，完整网格保留。TMoT列为v54 task256原版，不能代表v68因果参考扩展。','','| 数据 | 表示 | FT | 蒸馏 | 强度 | ER10监督头 | 原TMoT |','|---|---|---:|---:|---:|---:|---:|']
for ref in old:
 d,e=ref['dataset'],ref['encoder'];cl=ref['task']=='classification';group=[v for v in runs if (v['dataset'],v['encoder'])==(d,e)];ft=next(v for v in group if v['strength']==0);kd=sorted([v for v in group if v['strength']>0],key=lambda v:-v['final'] if cl else v['final'])[0];metric='macro_accuracy' if cl else 'rmse';h=ref['head'];head=dict(final=h['final_'+metric],average=h['average_'+metric],source=h['git_commit'],optimizer_steps=[v['optimizer_steps'] for v in h['stages']]);scale=100 if cl else 1
 rows.append(dict(dataset=d,encoder=e,task=ref['task'],ft=ft,distill=kd,er10_head=head,tmot=ref['tmot']))
 lines.append(f"| {d} | {e} | {ft['final']*scale:.3f} | {kd['final']*scale:.3f} | {kd['strength']:g} | {head['final']*scale:.3f} | {ref['tmot']['final']*scale:.3f} |")
lines+=['','## 解释与边界','','蒸馏在六个回归配置中均降低FT的最终RMSE，但最佳强度由开发VAL选择，不代表一个固定强度的普遍优势。Beijing+Mantis和Energy+Mantis优于ER10监督头，其他四个回归组合仍弱于该头；不能据此替代更强的累计ridge、FeCAM或因果参考基线。','','分类蒸馏改善很小，FT和蒸馏都严重遗忘。旧类子空间softmax对所有旧logit加同一个常数不变，因此旧类KL不能约束旧类整体相对新类的分数偏移。这是损失的可辨识性缺口；当前结果与此机制相容，但尚不能将全部失败归因于它。下一轮应直接检查跨新旧类的分数校准与稳定统计锚，而非只增加旧类KL权重。','','FT与蒸馏使用相同当前任务样本、epochs、batch和优化器更新步数；教师前向增加蒸馏计算。ER10每阶段额外抽取4096个历史代表，虽然epochs相同，但优化器步数更多，因此本轮与ER比较是统一架构/协议对照，不能称同计算预算因果消融。','','无历史样本缓存，教师仅在当前TRAIN输入上产生目标。分类为LwF思想的单头类增长适配；回归是输出MSE蒸馏，非原版LwF。参数继承与教师不变已测试。方法来源：https://arxiv.org/abs/1606.09282 。','','总体目标尚未完成；这些弱于FeCAM的对照不会取代强基线。分类和回归继续共同验证，不从这轮局部收益宣称一般时序持续学习的稳健优势。']
recency=Path('results/sequential_head_v73_recency.json')
if recency.exists():
 diag=json.loads(recency.read_text());lines+=['','## 最终阶段旧类诊断','','采用上述选定蒸馏强度，仅为预测行为诊断，不反馈训练。','','| 数据 | 表示 | 旧类宏准确率% | 旧样本判为最新类% | 最新类宏准确率% |','|---|---|---:|---:|---:|']
 for r in rows:
  if r['task']!='classification':continue
  a=next(v for v in diag if v['dataset']==r['dataset'] and v['encoder']==r['encoder'] and v['strength']==r['distill']['strength'])
  lines.append(f"| {r['dataset']} | {r['encoder']} | {100*a['old_macro']:.3f} | {100*a['old_predicted_as_new']:.3f} | {100*a['new_macro']:.3f} |")
Path('results/sequential_head_v73_comparison.json').write_text(json.dumps(rows,indent=2));Path('docs/SEQUENTIAL_HEAD_REPORT.md').write_text('\n'.join(lines)+'\n')
