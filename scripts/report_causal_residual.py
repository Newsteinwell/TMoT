import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
audit=json.loads((root/'results/causal_residual_v79_audit.json').read_text());assert audit['complete'] and audit['checkpoint_predictions_exact']
runs=audit['runs'];warm=json.loads((root/'results/warm_budget_v58_comparison.json').read_text())['rows'];causal=json.loads((root/'results/causal_task_v68_comparison.json').read_text())['selected']
lines=['# v79 强解析基线上的非线性残差：结果','', '## Material Passport','', '- 状态：48次训练、240阶段完成，最终模型预测按原GPU与相同分批逐元素复现。','- 来源：causal_residual_v79_audit.json；训练源版本 ab4441b052c13a4d234de9b5aba7379372a1e899。','- 范围：TRAIN/VAL 开发结果，未访问 TEST；不是独立确认或多种子验证。','', '## 与既有强基线比较','', '表中 local/ER10残差均按本轮最终VAL选最优，属于开发上界；不代表一套固定参数在所有数据上通过。强基线取既有因果ridge与warm ER10中更优的最终RMSE，不含任何本轮选择。','', '|数据|表示|既有强基线|local最佳（参考/惩罚）|local最终/平均|ER10残差最终/平均|local相对强基线改善|','|---|---|---:|---|---:|---:|---:|']
for d in ('air_quality','beijing_pm25'):
 for e in ('mantis','summary'):
  local=min((s for s in runs if (s['config']['dataset'],s['config']['encoder'],s['config']['mode'])==(d,e,'local')),key=lambda s:s['final_rmse'])
  er=min((s for s in runs if (s['config']['dataset'],s['config']['encoder'],s['config']['mode'])==(d,e,'er10')),key=lambda s:s['final_rmse'])
  old=min([r['warm_er10']['final'] for r in warm if (r['dataset'],r['encoder'])==(d,e)]+[s['final_rmse'] for s in causal if (s['config']['dataset'],s['config']['encoder'],s['config']['method'])==(d,e,'ridge')])
  c=local['config'];lines.append(f"|{d}|{e}|{old:.6f}|{c['reference']}/{c['penalty']:g}|{local['final_rmse']:.6f}/{local['average_rmse']:.6f}|{er['final_rmse']:.6f}/{er['average_rmse']:.6f}|{100*(old-local['final_rmse'])/old:.2f}%|")
lines+=['','## 全配置与训练诊断','', '训练诊断在最终模型训练结束后读取既有TRAIN做回顾评估，不参与优化。负改善表示残差恶化预测。','', '|数据/表示|参考|回放|惩罚|最终/平均RMSE|同输入base最终|旧任务TRAIN base→残差|当前TRAIN base→残差|','|---|---|---|---:|---:|---:|---:|---:|']
for s in runs:
 c=s['config'];t=s['retrospective_train_diagnostic'];lines.append(f"|{c['dataset']}/{c['encoder']}|{c['reference']}|{c['mode']}|{c['penalty']:g}|{s['final_rmse']:.4f}/{s['average_rmse']:.4f}|{s['stages'][-1]['base_rmse']:.4f}|{t['old']['base_rmse']:.4f}→{t['old']['residual_rmse']:.4f}|{t['current']['base_rmse']:.4f}→{t['current']['residual_rmse']:.4f}|")
lines+=['','## 复核与结论','', '全部240阶段指标由保存预测重算；解析分支与v68同配置预测逐元素一致。24组local/ER10首阶段配对预测完全一致。ER10逐任务缓存满足floor(N/10)，没有借额度；Air最终9个，Beijing最终78个。历史4096次抽样、优化步数、暖启动误差、输入哈希和模型哈希均复核。', '', '第一次验证使用CPU和全VAL批次，出现一个超出容差的1.203e-5差异。随后改用原GPU和相同分组推理，48个模型最终预测均精确一致；没有修改或重训模型，也没有放宽容差。首次失败日志单独保留。', '', '即使允许本轮逐数据选最优，四个数据/表示组合的local残差均弱于既有强基线；不能升级为默认方法。ER10残差也未解决这一差距，说明问题不能只归因于合成历史样本。解析强基线已经解释的部分不应被学习分支重新破坏。', '', 'local的24/24配置同时改善旧任务TRAIN和当前TRAIN，只有7/24改善同输入基线的最终VAL；ER10对应21/24、24/24和8/24。这支持训练收益未泛化的诊断，不能据此证明某一种过拟合机制。单纯延长训练缺少当前证据支持。', '', '下一步应区分残差过拟合、基线逐阶段变化导致旧残差失配和表示缺少新信息，例如在相同预算下比较残差冷启动/暖启动并保留强解析控制。当前证据不证明上述任何单一机制是唯一原因。类别新增的强FeCAM对照和一般多维时序目标保持不变；总体验收仍未通过。']
(root/'docs/CAUSAL_RESIDUAL_REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:20]))
