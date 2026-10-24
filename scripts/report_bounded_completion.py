import json
from pathlib import Path
rows=json.loads(Path('results/bounded_completion_v63_comparison.json').read_text())['rows'];audit=json.loads(Path('results/bounded_completion_v63_audit.json').read_text())
lines=['# v63 单位球约束后的完整类增长结果','', '源码d16e2f14a42550f308ef3a7d28c000fba817de9b；三个困难分类集×两种表示、seed42/order17，6个共享训练、72条完整轨迹。6项相关测试通过。只用原TRAIN/VAL，不访问TEST。当前任务数据进入模型，历史仅保留累计统计与合成点，不读取历史真实样本重拟合。','', 'complete为v62未约束校正；bounded在隐藏二阶范数超过1时以无新增超参数的方式同步缩放隐藏均值和相关协方差块，保持精确输入统计和半正定性。该必要约束不保证真实联合分布可恢复，也不保证预测改善。','', '首轮5个作业因诊断字段的 NumPy 布尔值无法序列化而失败，原日志保留；修复日志类型后在独立 retry 目录重跑六组，以下仅使用重跑完整结果。\n\n## 最终宏准确率（%）','', '| 数据集 | 表示 | 原TMoT | 未约束complete | bounded | 已有最强解析基线 |','|---|---|---:|---:|---:|---:|']
for r in rows:
 vals=[r['original_tmot_final'],r['modes']['complete']['final'],r['modes']['bounded']['final'],r['strongest_analytic_final']]
 lines.append('| '+r['dataset']+' | '+r['encoder']+' | '+' | '.join(f'{100*v:.3f}' for v in vals)+' |')
lines+=['','## 阶段平均宏准确率（%）','', '| 数据集 | 表示 | 原TMoT | 未约束complete | bounded |','|---|---|---:|---:|---:|']
for r in rows:
 vals=[r['original_tmot_average'],r['modes']['complete']['average'],r['modes']['bounded']['average']]
 lines.append('| '+r['dataset']+' | '+r['encoder']+' | '+' | '.join(f'{100*v:.3f}' for v in vals)+' |')
lines+=['','各模式均按同一六值alpha网格的最终VAL选择。下面在bounded所选alpha下成对比较，隔离收缩操作的影响，不混入不同alpha的差异。','', '| 数据集 | 表示 | alpha | bounded−complete最终（百分点） | bounded−complete阶段平均（百分点） |','|---|---|---:|---:|---:|']
for r in rows:
 b=r['modes']['bounded'];c=r['complete_at_bounded_alpha'];lines.append(f"| {r['dataset']} | {r['encoder']} | {b['alpha']:g} | {100*(b['final']-c['final']):+.3f} | {100*(b['average']-c['average']):+.3f} |")
geo=[r for r in audit['geometry'] if r['mode']=='bounded'];viol=sum(r['outside_unit_ball_moment_bound'] for r in geo)
lines+=['','## 核查','', f"72条完整轨迹指标核验，36条complete轨迹逐阶段复现v62；原TMoT训练状态与v54保持一致。最终预测用独立三角方程求解重算。独立残差重构协方差最大相对误差{audit['max_independent_covariance_relative_error']:.3g}。",'', f"输入均值与输入协方差逐项保持；最终{len(geo)}个类别中，bounded越过单位球矩上界的数量为{viol}，最大二阶范数{max(r['hidden_second_norm'] for r in geo):.12g}。检查针对alpha收缩之前的统计矩，不把数值正则化后的协方差当作真实分布矩。",'', '这仍是单种子、单顺序开发结果，不是稳健性验收或正式测试集结果。不按数据集事后拼接最优头，不将分类效果外推到回归目标。已有回归强基线和失败场景继续保留；总体目标仍为一般多维时间序列持续学习。']
Path('docs/BOUNDED_COMPLETION_REPORT.md').write_text('\n'.join(lines)+'\n')
print('wins over strongest analytic:',sum(r['modes']['bounded']['final']>r['strongest_analytic_final']+1e-12 for r in rows))
