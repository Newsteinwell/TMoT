import json
from pathlib import Path
r=json.loads(Path('results/prequential_reference_v70_comparison.json').read_text());old=json.loads(Path('results/causal_task_v68_comparison.json').read_text());assert r['complete']
lines=['# v70 训练前误差加权的三先验融合','', '源码4d86f361bc3485df124f364416810d643437056f；4个数据/表示组、12个神经专家训练、96条完整融合轨迹。两项权重测试和五阶段烟测通过。旧TRAIN/VAL、seed42，无TEST；六值ridge按最终VAL选择，非独立盲测。','', '三个专家为none、append、residual，目标坐标分别固定。第一任务等权；后续每个任务先预测当前TRAIN、累计平方误差，再训练当前任务，最后评估VAL。逆累计误差作为权重；所有专家真实训练样本数相同。另有始终等权对照，ridge获得相同机制。','', '| 数据集 | 表示 | 融合 | TMoT最终RMSE | Ridge最终RMSE | TMoT阶段平均 | Ridge阶段平均 |','|---|---|---|---:|---:|---:|---:|']
for d,e,p in sorted({tuple(s['config'][k] for k in ('dataset','encoder','policy')) for s in r['selected']}):
 pair={s['config']['method']:s for s in r['selected'] if tuple(s['config'][k] for k in ('dataset','encoder','policy'))==(d,e,p)};t,b=pair['task'],pair['ridge'];lines.append(f"| {d} | {e} | {p} | {t['final_rmse']:.4f} | {b['final_rmse']:.4f} | {t['average_rmse']:.4f} | {b['average_rmse']:.4f} |")
lines+=['','## 保留单专家强参考','', '以下单专家按旧开发VAL选参，仅作已知强参考，不能把事后选择三个先验之一当作新算法。','', '| 数据集 | 表示 | 旧最佳TMoT先验 | 最终RMSE | 旧最佳Ridge先验 | 最终RMSE |','|---|---|---|---:|---|---:|']
for d,e in sorted({(s['config']['dataset'],s['config']['encoder']) for s in r['selected']}):
 pair={m:min((s for s in old['selected'] if (s['config']['dataset'],s['config']['encoder'],s['config']['method'])==(d,e,m)),key=lambda s:s['final_rmse']) for m in ('task','ridge')};t,b=pair['task'],pair['ridge'];lines.append(f"| {d} | {e} | {t['config']['reference']} | {t['final_rmse']:.4f} | {b['config']['reference']} | {b['final_rmse']:.4f} |")
lines+=['','## 权重与成本','', '| 数据集 | 表示 | 方法 | 最优ridge | 最终none/append/residual权重 |','|---|---|---|---:|---|']
for s in r['selected']:
 c=s['config']
 if c['policy']=='prequential':lines.append(f"| {c['dataset']} | {c['encoder']} | {c['method']} | {c['ridge']:g} | "+' / '.join(f'{w:.4f}' for w in s['stages'][-1]['weights'])+' |')
lines+=['', '三分支约为单分支三倍训练与模型状态成本；不能只报告权重状态很小而忽略专家本身。离线保存训练前预测/ID/标签用于核验，不进入回放或在线权重状态，在线只维护三个SSE和计数。','', '全部96条轨迹从原始训练前预测独立重算累计SSE、权重和混合预测；各阶段专家预测精确复现v68，样本身份、各阶段RMSE和输入SHA核验通过。预测先于当前任务训练由实验程序的循环顺序保证；没有使用VAL标签更新权重。','', '本轮针对具有历史目标的回归预测，不能将结果外推类增长分类或任意连续标签回归；一般多维时序持续学习总体目标保持不变。']
Path('docs/PREQUENTIAL_REFERENCE_REPORT.md').write_text('\n'.join(lines)+'\n')
