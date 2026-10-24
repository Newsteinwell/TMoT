import json
from pathlib import Path
folder=Path('results/support_diagnostic_v76');rows=[]
for p in folder.glob('*.json'):
 s=json.loads(p.read_text())
 if isinstance(s,dict):rows.extend(s.get('rows',[]))
assert len(rows)==36
lines=['# v76 支持输入与教师目标诊断','','只读v75最终检查点，12个数据/表示作业、36个支持配置全部完成。支持输入逐值重建，哈希与训练记录一致；检查点SHA前后不变。没有训练更新，没有增加真实样本缓存。','','下表为分类最后阶段的支持输入。生成分量标签是名义来源，不能把教师与它的一致率等同于真实测试准确率。熵除以log(已见类别数)，1表示均匀分布。','','| 数据 | 表示 | 支持 | 最大概率均值 | 归一化熵 | 教师-名义标签一致率% | 学生-教师一致率% | 名义类别数/已见类数 |','|---|---|---|---:|---:|---:|---:|---:|']
for r in sorted(rows,key=lambda r:(r['dataset'],r['encoder'],r['support'])):
 if r['task']=='classification':lines.append(f"| {r['dataset']} | {r['encoder']} | {r['support']} | {r['mean_max_probability']:.4f} | {r['normalized_entropy']:.4f} | {100*r['teacher_nominal_agreement']:.2f} | {100*r['student_teacher_agreement']:.2f} | {r['nominal_classes']}/{r['seen_classes']} |")
lines+=['','local支持覆盖所有历史类别（不含最后到达的新类）。其目标熵0.976–0.997，Phoneme+summary最大概率均值仅0.0344，39类均匀概率为0.0256；学生在同一支持上的教师一致率仅10.77%。这支持进一步检验软目标过平的问题，但不能证明温度是唯一原因。','','Handwriting生成支持上教师-名义标签一致率100%，但不能说明它覆盖了真实VAL的变化；不同支持分布、目标置信度和学生拟合误差仍需分开验证。','','回归local支持上学生-教师RMSE：Air两表示0.175/0.173，Beijing 5.438/3.126，Energy 7.339/5.275。对应教师-生成名义目标RMSE更高。它们都是支持分布上的量，不是最终测试误差，也不能直接证明真实数据不可约噪声。全部36配置见原始JSON。','','因此v77固定比较原软目标、温度0.1目标与硬目标，保留教师argmax和训练输入不变；回归目标不变并重跑控制。']
Path('docs/SUPPORT_DIAGNOSTIC_REPORT.md').write_text('\n'.join(lines)+'\n')
audit=Path('results/target_confidence_v77_audit.json')
if not audit.exists():raise SystemExit(0)
runs=json.loads(audit.read_text())['runs'];assert len(runs)==24
lines=['# v77 教师目标置信度消融','','训练源码273e92e14dba14149fd117a948bae6c9291bd7a1。18条分类轨迹加6条回归控制，7项相关测试通过。固定local支持与lambda10，仅改变分类教师目标，详见协议；没有TEST或本轮参数网格扩展。','','soft完整轨迹精确复现v75 local，包括六个回归控制；各模式支持SHA、教师预测及优化器步数相同。全部阶段指标、样本身份和最终模型预测核验通过。','','分类为最终/阶段平均宏准确率百分数。固定教师为alpha10 FeCAM，不能等同各数据最优选参基线。','','| 数据 | 表示 | soft | sharp(T=0.1) | hard | 直接教师 |','|---|---|---:|---:|---:|---:|']
for d,e in sorted(set((r['dataset'],r['encoder']) for r in runs if r['task']=='classification')):
 g={r['target_mode']:r for r in runs if r['dataset']==d and r['encoder']==e}
 def fmt(m):return f"{g[m]['final']*100:.3f}/{g[m]['average']*100:.3f}"
 t=g['soft'];lines.append(f"| {d} | {e} | {fmt('soft')} | {fmt('sharp')} | {fmt('hard')} | {t['teacher_final']*100:.3f}/{t['teacher_average']*100:.3f} |")
lines+=['','六个回归控制结果未改变，见v75报告。分类目标锐化结果不外推回归；整个框架仍需两类任务共同验证。新结果是单种子/类序的开发证据，不是稳健泛化证明。']
lines+=['','## 与既有强读出比较','','下表列各目标形式中最终VAL最高值，仅用于诊断距离强基线还有多远，不把这种事后挑选拼成算法。强基线列为v60中既有FeCAM及全部解析读出的最大最终值。','','| 数据 | 表示 | 本轮最佳最终% | 既有强读出最终% | 差值pp |','|---|---|---:|---:|---:|']
strong=json.loads(Path('results/analytic_heads_v60_comparison.json').read_text())['rows']
for r in strong:
 d,e=r['dataset'],r['encoder'];g=[v for v in runs if v['dataset']==d and v['encoder']==e]
 best=max(v['final'] for v in g);baseline=max([r['fecam_final']]+[v['final'] for v in r['heads'].values()]);lines.append(f"| {d} | {e} | {100*best:.3f} | {100*baseline:.3f} | {100*(best-baseline):+.3f} |")
lines+=['','## 结论','','锐化相对soft在四个分类配置改善最终准确率，一个持平、一个退化；硬目标在五个改善、一个退化。LSST+Mantis的锐化改善13.342个百分点，Phoneme+Mantis硬目标改善13.876个百分点。教师argmax、支持输入和更新步数没有变化，因此这是目标形式对当前训练流程的影响，而不是新增教师判别信息。','','本轮最佳结果在六个配置仍全部低于既有强读出。Handwriting+Mantis硬目标40.385%虽高于固定FeCAM 36.538%，但其阶段平均32.822%低于教师36.026%，且低于既有ridge最终51.923%；不能称稳健胜出。其余五个配置即使择优也低于固定教师。','','v75和v77共同表明历史覆盖与软目标平缓都是可改善环节，但没有证明学习模型能够超越强统计读出。保持统计教师的直接推断能力，后续优先验证新增表示/统计信息是否形成互补，而不是继续把一个弱于强基线的模仿模型作为默认预测器。本轮模块不升级为默认TMoT；一般多维时序持续学习目标与分类、回归共同验收要求保持不变。']
Path('docs/TARGET_CONFIDENCE_REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[6:]))
