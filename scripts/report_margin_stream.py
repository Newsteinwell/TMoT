import json
from pathlib import Path
r=json.loads(Path('results/margin_stream_v67_comparison.json').read_text());assert r['complete']
lines=['# v67 相对间隔选择的完整任务流复验','', '训练源码8a38d119ae7738b9afffd28b0c8a8ac6df474128。36组共享训练、216条完整轨迹。规则和所有头参数固定，无新增阈值；原五个头逐阶段复现v64，最终距离和选择精确复现v66。端到端Handwriting烟测通过，规则的三项测试见v66。','', '每行平均除seed42/order17控制外的五个组合；控制和所有单次结果保存在JSON。重复使用开发TRAIN/VAL，不是独立盲测；无TEST。种子/顺序组合不是独立数据集。单位为宏准确率百分点差。','', '| 数据集 | 表示 | 参考 | 最终均值差 | 阶段平均差 | 阶段平均差范围 | 最终胜出/5 | 阶段平均胜出/5 |','|---|---|---|---:|---:|---:|---:|---:|']
for s in r['summary']:
 lines.append(f"| {s['dataset']} | {s['encoder']} | {s['reference']} | {s['mean_final_gain_pp']:+.3f} | {s['mean_average_gain_pp']:+.3f} | [{s['min_average_gain_pp']:+.3f}, {s['max_average_gain_pp']:+.3f}] | {s['final_wins']} | {s['average_wins']} |")
lines+=['','## 旧任务遗忘与后向迁移','', '遗忘：各旧任务从学到该任务起至最终阶段的最高准确率减最终准确率，含最终阶段所以非负；BWT：最终准确率减该任务刚学完时准确率。对旧任务等权平均，排除最后任务。以下为相对参考的遗忘减少量与BWT增量，正值更好。','', '| 数据集 | 表示 | 参考 | 遗忘减少（pp） | BWT增量（pp） |','|---|---|---|---:|---:|']
for s in r['summary']:
 lines.append(f"| {s['dataset']} | {s['encoder']} | {s['reference']} | {s['mean_forgetting_reduction_pp']:+.3f} | {s['mean_backward_transfer_gain_pp']:+.3f} |")
lines+=['','## 核验与范围','', '每阶段从保存的全部类别距离独立排序重算选择标志及预测；核对两分支原预测、输入ID、标签、各任务与总体宏准确率。最终选择和距离矩阵复现v66；所有原五头预测复现v64。冻结参数、输入哈希与复用检查点SHA逐项核对。训练结束时网络、g/k、合成点和RNG精确复现v55，单位化统计精确复现v64。','', '选择函数不访问评估标签。训练只使用当前任务真实样本、历史累计统计和合成点；保存的VAL距离仅用于核验。引用旧模型避免重复磁盘副本，并不代表算法没有内存成本。','', '总体目标仍为一般多维时序持续学习。本轮只检验类增长条件，不将分类间隔规则直接用于连续标签或宣称回归有效。']
Path('docs/MARGIN_STREAM_REPORT.md').write_text('\n'.join(lines)+'\n')
