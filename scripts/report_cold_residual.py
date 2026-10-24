"""Describe the fixed-budget cold/warm ablation without treating configs as datasets."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
cold=json.loads((root/'results/cold_residual_v80_audit.json').read_text())
warm=json.loads((root/'results/causal_residual_v79_audit.json').read_text())
assert cold['complete'] and cold['checkpoint_predictions_exact'] and len(cold['runs'])==48
def key(s):
 c=s['config'];return tuple(c[k] for k in ('dataset','encoder','reference','mode','penalty','ridge'))
w={key(s):s for s in warm['runs']};pairs=[(s,w[key(s)]) for s in cold['runs']]
assert len(pairs)==48
lines=['# v80 冷启动/暖启动残差对照','', '## Material Passport','', '- 状态：48个冷启动训练完成，240阶段复核通过；对应v79暖启动48配置。','- 数据：复用开发TRAIN/VAL；不读取TEST，不是独立确认。','- 训练源版本：3909cdf5d4440970b4ea0685a2edc207d80cf86c。','', '## 固定预算配对结果','', '|回放|冷启动最终改善/24|冷启动阶段平均改善/24|两个指标同时改善/24|','|---|---:|---:|---:|']
for mode in ('local','er10'):
 ps=[(c,w) for c,w in pairs if c['config']['mode']==mode]
 final=sum(c['final_rmse']<w['final_rmse'] for c,w in ps);avg=sum(c['average_rmse']<w['average_rmse'] for c,w in ps);both=sum(c['final_rmse']<w['final_rmse'] and c['average_rmse']<w['average_rmse'] for c,w in ps)
 lines.append(f'|{mode}|{final}|{avg}|{both}|')
lines+=['','配置高度相关，上表不作为48个独立数据集或统计显著性证据。','', '## 与强基线的距离','', '以下最优值按本轮最终VAL选择，仅是开发上界，不是固定策略的盲测结果。','', '|数据|表示|既有强基线|暖local最佳|冷local最佳|冷local平均|冷local配置|冷ER10最佳|','|---|---|---:|---:|---:|---:|---|---:|']
warm_old=json.loads((root/'results/warm_budget_v58_comparison.json').read_text())['rows'];causal=json.loads((root/'results/causal_task_v68_comparison.json').read_text())['selected']
for d in ('air_quality','beijing_pm25'):
 for e in ('mantis','summary'):
  rows=[s for s in cold['runs'] if (s['config']['dataset'],s['config']['encoder'])==(d,e)]
  c=min((s for s in rows if s['config']['mode']=='local'),key=lambda s:s['final_rmse']);er=min((s for s in rows if s['config']['mode']=='er10'),key=lambda s:s['final_rmse'])
  wb=min(s['final_rmse'] for s in warm['runs'] if (s['config']['dataset'],s['config']['encoder'],s['config']['mode'])==(d,e,'local'))
  base=min([s['warm_er10']['final'] for s in warm_old if (s['dataset'],s['encoder'])==(d,e)]+[s['final_rmse'] for s in causal if (s['config']['dataset'],s['config']['encoder'],s['config']['method'])==(d,e,'ridge')])
  lines.append(f"|{d}|{e}|{base:.6f}|{wb:.6f}|{c['final_rmse']:.6f}|{c['average_rmse']:.6f}|{c['config']['reference']}/p{c['config']['penalty']:g}|{er['final_rmse']:.6f}|")
lines+=['','## 完整配对','', '|数据/表示|参考|回放|惩罚|暖最终/平均|冷最终/平均|冷旧TRAIN base→残差|冷当前TRAIN base→残差|','|---|---|---|---:|---:|---:|---:|---:|']
for c,w in pairs:
 a=c['config'];t=c['retrospective_train_diagnostic']
 lines.append(f"|{a['dataset']}/{a['encoder']}|{a['reference']}|{a['mode']}|{a['penalty']:g}|{w['final_rmse']:.4f}/{w['average_rmse']:.4f}|{c['final_rmse']:.4f}/{c['average_rmse']:.4f}|{t['old']['base_rmse']:.4f}→{t['old']['residual_rmse']:.4f}|{t['current']['base_rmse']:.4f}→{t['current']['residual_rmse']:.4f}|")
lines+=['','## 核验','', '48个首阶段预测与暖启动逐元素一致；240阶段解析预测、标签、ID、历史支持数量、优化步数以及ER10选样索引一致。240阶段指标由保存预测重算，48个最终模型按原GPU和分批精确复现。输入与检查点哈希通过。训练诊断在训练结束后进行，没有用于训练。']
lines+=['','## 诊断与决策','', '冷启动local在14/24配置中改善最终RMSE，但最终与阶段平均同时改善的只有10/24；ER10对应8/24与5/24。因此不能将冷启动定为普遍更优默认。四个数据/表示组合即使选本轮最佳local，仍全部弱于既有强基线。', '', '冷启动local的20/24配置降低旧任务TRAIN误差，24/24降低当前TRAIN误差，但仅7/24降低同输入base的最终VAL误差。ER10对应2/24、24/24、5/24；其暖启动旧TRAIN改善为21/24。这说明网络参数本身携带了缓存之外的历史信息，冷启动会丢失这部分信息。这个解释与观察相容，但不是独立因果证明。', '', '单独取消暖启动不能解决泛化差距；本轮不采用冷启动为默认，也不继续仅凭初始化或增加训练轮数搜索优势。应将下一阶段重点转回表示是否缺少跨通道/时间交互信息，以及学习分支如何在训练内验证后保留强统计读出的收益。新增表示须同时给强基线使用，避免把输入优势误称为持续学习算法优势。分类类别新增和回归仍共同纳入总体验收，当前目标未完成。']
(root/'docs/COLD_RESIDUAL_REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:30]))
