import json
from pathlib import Path
rows=json.loads(Path('results/class_completion_v62_comparison.json').read_text())['rows'];audit=json.loads(Path('results/class_completion_v62_audit.json').read_text())
lines=['# v62 原始表示归一化统计校正：完整任务流结果','', '源码ba17d45d03a286f5b9e9b16e49671a81d76e5ce7；三个困难分类集×两种表示、seed42/order17，6个共享训练作业、144条完整类增长轨迹。仅原TRAIN/VAL，不访问TEST。网络训练与v54 task256完全相同；新读出只用当前任务数据、累计统计和已保存合成点，不读取历史真实样本。','', 'base：精确unit(phi)统计距离头；approx：合成点分块归一化联合统计；self：正则化条件投影后代回原合成输入统计；complete：代入精确累计unit(phi)统计。每种模式按同一六值alpha网格的最终VAL选择，同时报告阶段平均指标。投影rho固定.01，不额外调参。','', '## 最终宏准确率（%）','', '| 数据集 | 表示 | 原TMoT | base | approx | self | complete | 已有最强解析基线 |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
 vals=[r['original_tmot_final']]+[r['modes'][m]['final'] for m in ('base','approx','self','complete')]+[r['strongest_analytic_final']]
 lines.append('| '+r['dataset']+' | '+r['encoder']+' | '+' | '.join(f'{100*v:.3f}' for v in vals)+' |')
lines+=['','最强解析参考列仅列出已有方法中的最好最终结果，不把多个方法合成为一个新方法。具体名称在比较JSON中。','', '## 阶段平均宏准确率（%）','', '| 数据集 | 表示 | 原TMoT | base | approx | self | complete |','|---|---|---:|---:|---:|---:|---:|']
for r in rows:
 vals=[r['original_tmot_average']]+[r['modes'][m]['average'] for m in ('base','approx','self','complete')]
 lines.append('| '+r['dataset']+' | '+r['encoder']+' | '+' | '.join(f'{100*v:.3f}' for v in vals)+' |')
lines+=['','## 统计校正的独立影响','', '同一alpha下complete与self的差异才隔离精确输入统计的作用；self与approx的差异包含正则化投影舍弃残差相关项的影响。不能把complete相对approx的全部增益归因于精确统计。','', '| 数据集 | 表示 | complete−self同alpha最终（百分点） | complete−self同alpha阶段平均（百分点） | complete alpha | complete持久数组MiB |','|---|---|---:|---:|---:|---:|']
for r in rows:
 c=r['modes']['complete'];s=r['self_at_complete_alpha'];lines.append(f"| {r['dataset']} | {r['encoder']} | {100*(c['final']-s['final']):+.3f} | {100*(c['average']-s['average']):+.3f} | {c['alpha']:g} | {c['persistent_arrays_bytes']/2**20:.2f} |")
lines+=['','## 额外几何诊断','', '以下检查的是读出alpha收缩前的统计矩。归一化隐藏向量的平方范数至多为1，因此其预测二阶范数大于1说明统计校正结果不能严格解释为这种向量的真实矩；这不等同于分类预测必然错误。该检查是本轮附加诊断，不参与参数选择。','', '| 数据集 | 表示 | complete越界类别数/总数 | complete最大二阶范数 |','|---|---|---:|---:|']
for r in rows:
 g=[v for v in audit['geometry'] if (v['dataset'],v['encoder'],v['mode'])==(r['dataset'],r['encoder'],'complete')]
 lines.append(f"| {r['dataset']} | {r['encoder']} | {sum(v['outside_unit_ball_moment_bound'] for v in g)}/{len(g)} | {max(v['hidden_second_norm'] for v in g):.6f} |")
lines+=['','## 核查与限制','', f"3项新增测试通过，覆盖精确输入块、半正定性、无正则情况下的恒等重建、Cholesky/FeCAM逆矩阵一致性、连续任务时不修改训练状态。144条轨迹指标和最终预测核查通过；36条base轨迹逐阶段复现FeCAM。独立直接残差样本重构校正协方差，最大相对误差{audit['max_independent_covariance_relative_error']:.3g}。",'', '各作业还逐阶段复现原TMoT预测，最终网络参数、g/k、合成点和主RNG与v54完全一致。新增单位化输入矩从当前任务逐步累计，最终与直接TRAIN统计核对；原始检查点通过SHA引用，避免重复保存。','', '本模块增加每类单位化输入二阶矩，内存随类别数增长，不是固定总内存方法。表中持久数组含同一TMoT训练流水线及新统计、对应头矩统计，不是独立FeCAM最小内存，也不包含临时工作区/Python对象。','', '这是单种子单顺序的开发实验；不能将条件间事后最优头拼成统一方法，也不能以个别数据集收益声称普遍优势。本轮为分类模块实验，回归目标和已有强基线保留，仍需在后续回归任务验证任何通用改法。']
Path('docs/CLASS_COMPLETION_REPORT.md').write_text('\n'.join(lines)+'\n')
print('complete wins versus self:',sum(r['modes']['complete']['final']>r['modes']['self']['final']+1e-12 for r in rows))
print('complete wins versus strongest analytic:',sum(r['modes']['complete']['final']>r['strongest_analytic_final']+1e-12 for r in rows))
