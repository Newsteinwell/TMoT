import json
from pathlib import Path
r=json.loads(Path('results/bounded_robustness_v64_comparison.json').read_text());a=json.loads(Path('results/bounded_robustness_v64_audit.json').read_text())
lines=['# v64 固定参数跨种子与任务顺序复验','', '训练源码a4546c8da6a21d40bc1fa1d754de827c5ba0aeda。36次共享训练、180条完整分类轨迹。三个困难数据集、两种表示、seed42/43/44与order17/23。所有头的超参数在seed42/order17选定后冻结；不在其他组合重新选择。无TEST。','', 'seed42/order17作为控制不计入下表；每行描述其余五个组合。这些组合此前已用于v55开发，因此不是独立盲测。重复种子/顺序并非独立数据集，不将本表作为显著性检验。','', '## bounded 相对各参考方法的百分点差','', '| 数据集 | 表示 | 参考 | 最终均值差 | 最终差范围 | 阶段平均差 | 最终胜出/5 | 最终≥1pp/5 |','|---|---|---|---:|---:|---:|---:|---:|']
for s in r['summary']:
 lines.append(f"| {s['dataset']} | {s['encoder']} | {s['reference']} | {s['mean_final_gain_pp']:+.3f} | [{s['min_final_gain_pp']:+.3f}, {s['max_final_gain_pp']:+.3f}] | {s['mean_average_gain_pp']:+.3f} | {s['final_positive']} | {s['final_at_least_1pp']} |")
lines+=['','## 核验范围','',f"{a['trajectories']}条轨迹完成；36条原TMoT控制逐阶段复现v55；原开发配置的complete/bounded/FeCAM逐阶段复现v63/v59。核验固定参数、输入哈希、样本身份与每阶段宏准确率；最终距离头预测由独立三角求解重算，累计ridge由原TRAIN重建正规方程核验。独立残差协方差重建最大相对误差{a['max_independent_covariance_relative_error']:.3g}。",'', '训练时每阶段只访问当前任务数据与保留统计/合成点。审计从TRAIN重算仅用于核验，结果不反馈训练。新增单位化统计单独保存，原模型引用通过SHA确认；不将省去重复磁盘副本等同于算法降低内存。','', '全部结果保留，不事后按数据集/种子拼接不同头。分类结果不外推回归；一般多维时序持续学习目标不变，回归仍需相应验证。']
Path('docs/BOUNDED_ROBUSTNESS_REPORT.md').write_text('\n'.join(lines)+'\n')
