import json
from pathlib import Path
runs=json.loads(Path('results/support_teacher_v75_audit.json').read_text())['runs'];rows=[]
lines=['# v75 相同更新步数下的历史输入覆盖对照','','训练源码ddaa60420a13e01725fc268b36e85d54726d1244。36条完整轨迹，6个既有开发数据×2表示×3种支持方式。参数事先固定，无本轮超参数选择，无TEST。10项相关单元测试通过，另单独复测3项支持测试通过。','','所有首阶段三组预测完全一致；所有阶段教师预测逐值复现v74。独立核对各阶段指标、输入哈希、样本身份与最终学生检查点预测，重建支持抽样SHA、真实缓存选择索引和逐任务10%配额。三组每阶段优化器步数均为30×ceil((N_current+4096)/256)。','','分类为宏准确率百分数（越高越好），回归为RMSE（越低越好）；每格为最终/阶段平均。local为历史局部统计生成，er10为受配额限制的真实历史输入；两者历史标签均不参与监督损失，只接受固定教师约束。','','| 数据 | 表示 | 当前重采样 | local | er10 | 固定教师 |','|---|---|---:|---:|---:|---:|']
for d,e in sorted(set((r['dataset'],r['encoder']) for r in runs)):
 g={r['support']:r for r in runs if (r['dataset'],r['encoder'])==(d,e)};cl=g['current']['task']=='classification';scale=100 if cl else 1
 def fmt(mode):return f"{g[mode]['final']*scale:.3f}/{g[mode]['average']*scale:.3f}"
 c=g['current'];lines.append(f"| {d} | {e} | {fmt('current')} | {fmt('local')} | {fmt('er10')} | {c['teacher_final']*scale:.3f}/{c['teacher_average']*scale:.3f} |")
 rows.append(dict(dataset=d,encoder=e,task=c['task'],runs=g))
lines+=['','## 旧类别与资源','','| 数据 | 表示 | 当前旧类准确率% | local旧类准确率% | er10旧类准确率% | er10缓存数 | local额外存储MiB |','|---|---|---:|---:|---:|---:|---:|']
for r in rows:
 g=r['runs']
 if r['task']=='classification':lines.append(f"| {r['dataset']} | {r['encoder']} | {g['current']['old_macro']*100:.3f} | {g['local']['old_macro']*100:.3f} | {g['er10']['old_macro']*100:.3f} | {g['er10']['stored_examples']} | {g['local']['support_bytes']/2**20:.3f} |")
lines+=['','三者只匹配学生训练行数、batch和更新步数；统计生成、教师预测和实际运行时间不同，存储预算也不同。真实缓存每个历史任务最多floor(N/10)，累计随任务增长；零配额不能补借。首任务以及历史真实缓存为空时，支持输入回退到当前重采样，已经显式记录。','','统计教师固定alpha10/ridge0.1，不等于各数据选参后的最强基线。无论学生是否超过本轮教师，都需与既有完整强基线及TMoT结果比较。新方法未经过独立盲测或多种子复验，不据开发轨迹宣称普遍稳健优势。']
lines+=['','## 结论与后续机制验证','','在同一统计教师、网络、样本行数与更新步数下，local相对current在六个分类配置全部提高最终和阶段平均准确率；其中Handwriting+Mantis最终从3.846%到34.615%，LSST+Mantis从13.768%到44.381%。旧类准确率同时大幅恢复，支持“仅在当前输入约束学生，历史输入覆盖不足”这一诊断。这里仍有输入分布和正则化作用共同变化，不能解释为已证明唯一失效原因。','','local在六个分类配置也优于本轮er10输入支持，但它们存储预算不同，且er10只在历史输入上使用教师目标，并非使用历史真实标签的完整ER；不能据此宣称超过ER方法。Handwriting累计缓存仅6个样本，部分历史任务为零配额，这会限制历史类别覆盖。','','所有分类学生仍弱于固定FeCAM教师，说明恢复历史覆盖不等于无损传递其决策边界，更不等于新增判别能力。回归local相对current四个配置改善、两个退化；虽然六个配置最终RMSE都低于固定ridge教师，但该教师未做各数据最优选择，且阶段平均并非全部改善。不能替代既有强基线比较，也没有达到一般场景稳健优势。','','后续应检查支持输入上教师软目标的置信度和类别覆盖，区分支持分布误差与蒸馏目标过平导致的信息损失。优先保留稳定统计读出的直接预测能力，再验证学习分支能提供何种互补增益；不继续用一个明显更弱的学习头完全替代强读出。']
Path('results/support_teacher_v75_comparison.json').write_text(json.dumps(rows,indent=2));Path('docs/SUPPORT_TEACHER_REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[6:21]))
