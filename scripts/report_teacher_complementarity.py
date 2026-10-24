import json
from pathlib import Path
s=json.loads(Path('results/teacher_complementarity_v78.json').read_text());rows=s['rows'];assert len(rows)==70 and s['source_files_unchanged']
lines=['# v78 强读出与新学习模型互补性','','70对完整开发轨迹重放完成（54分类、16回归），4项分解测试通过。没有新训练、没有TEST。所有源预测SHA前后不变，样本身份/真实标签对齐，候选原指标按源dtype精确复算。首次float32/float64精度检查中止日志保留，retry源码c1a1d1eb6872debfd0fc5bb976a1841a1969eb61；残差分解统一float64并记录与原精度差。','','## 分类相对较强参照的纠错与破坏','','每个数据/表示的参照，在此前已选定的FeCAM与raw ridge中取较高最终准确率者，所有候选使用同一参照。表中均为最终类平衡百分点/百分数。纠错=参照错而候选对；破坏=参照对而候选错。oracle仅表示用真实标签在两套硬预测间完美选择的诊断上限，不能实现，也不是任意分数融合的上界。','','| 数据 | 表示 | 目标 | 参照 | 参照准确率% | 候选准确率% | 纠错pp | 破坏pp | oracle% |','|---|---|---|---|---:|---:|---:|---:|---:|']
selected=[]
for d,e in sorted(set((r['dataset'],r['encoder']) for r in rows if r['task']=='classification')):
 group=[r for r in rows if r['dataset']==d and r['encoder']==e and r['reference'] in ('selected_fecam','selected_raw_ridge')];ref=max(group,key=lambda r:r['stages'][-1]['reference_accuracy'])['reference']
 for r in group:
  if r['reference']!=ref:continue
  v=r['stages'][-1];selected.append(r);lines.append(f"| {d} | {e} | {r['mode']} | {ref} | {100*v['reference_accuracy']:.3f} | {100*v['candidate_accuracy']:.3f} | {100*v['rescue']:.3f} | {100*v['harm']:.3f} | {100*v['label_oracle_selector_accuracy']:.3f} |")
lines+=['','## 回归固定凸组合','','权重为学习模型占比，0是参照、1是学习模型。0.25/0.5/0.75在运行前固定；每格为最终/阶段平均RMSE，低为好。Oracle权重只由最终VAL标签求得，仅诊断，不用于任何实际组合。','','| 数据 | 表示 | 参照 | 参照RMSE | 学习模型RMSE | 0.25 | 0.5 | 0.75 | oracle权重 | oracle最终RMSE |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 if r['task']!='regression':continue
 v=r['stages'][-1]
 def fmt(key):return f"{v[key]:.4f}/{sum(t[key] for t in r['stages'])/len(r['stages']):.4f}"
 def mix(w):return f"{v['fixed_mixtures'][w]:.4f}/{sum(t['fixed_mixtures'][w] for t in r['stages'])/len(r['stages']):.4f}"
 lines.append(f"| {r['dataset']} | {r['encoder']} | {r['reference']} | {fmt('reference_rmse')} | {fmt('candidate_rmse')} | {mix('0.25')} | {mix('0.5')} | {mix('0.75')} | {v['label_oracle_beta']:.4f} | {v['label_oracle_rmse']:.4f} |")
lines+=['','固定混合的所有阶段权重相同，未用VAL选择门控；若随后从三个权重中选择最好者，则属于进一步开发选择，不能算独立确认。源方法此前已使用开发VAL选参，本轮也不能称盲测。完整结果保留所有参照和模式，不能把每个数据最有利的配对拼成通用方法。']
lines+=['','## 决策结论','','分类存在互补但并非无代价。Phoneme+Mantis硬目标相对选参FeCAM纠错5.430pp、破坏8.446pp；LSST+summary锐化纠错7.935pp、破坏10.592pp。不能把纠错量或oracle当成可实现收益。Handwriting+summary三种目标均无纠错，即使利用真实标签在两套硬预测中选择，也不能超过参照；这排除了在该配置继续优化两套硬标签选择器的理由，但不排除改变表示或分数融合产生新预测。','','固定回归混合在旧开发Energy+Mantis上有局部收益：学生权重0.75时最终RMSE64.046，低于warm ER10的69.692。但Energy不属于新增的正式回归验收集，不能用这一结果替代Air/Beijing证据；还须比较已有更强方法和相同总资源。','','Air+summary相对warm ER10最优单权重oracle为0.8033，对照0.8073，仅约0.50%改善；预先固定0.25权重为0.8047。Air+Mantis对应oracle约1.13%改善。Beijing+Mantis相对因果ridge的oracle选学生权重0，即无改善；Beijing+summary仅约0.18%。在当前这些固定预测上，即使标签oracle也达不到5%门槛，说明继续搜索单一凸组合权重不能解决正式回归验收差距。该判断仅针对当前预测的单权重凸组合，不外推任意非线性融合。','','因此不将固定混合升级为默认TMoT。下一步将学习分支直接建立在更强的稳定/因果统计预测之上，检验剩余误差是否存在可学习结构，并区分新增有效信息与对弱教师的重复拟合。分类保留强读出，同时需要验证可靠互补信息的来源，不能用VAL标签拟合oracle门控。总体目标仍未完成。']
Path('docs/TEACHER_COMPLEMENTARITY_REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[8:]))
