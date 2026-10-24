import json
from pathlib import Path
selected=json.loads(Path('results/class_distance_v61_selected.json').read_text());lines=['# 冻结TMoT后的类别距离读出诊断 v61','', '源码240b31cd053da781c1e0990fb28b0aeca5e8d0a0，三个困难分类数据集×两种表示，6个作业、216组最终阶段诊断，seed42/order17，只使用原TRAIN/VAL。未更新原模型，未访问TEST。每种条件在同一六值alpha网格中按最终VAL选参。','', '真实统计分支可访问全部已到达TRAIN，仅为离线诊断，不是满足缓存限制的持续学习方法，也不是理论性能上界。合成统计分支只读取保存的点，按单元原样本数加权。三个空间分别为原始512维input、学习隐藏128维hidden、拼接640维joint；全部逐样本L2单位化。','', '## 最终宏准确率（%）','', '| 数据集 | 表示 | 原TMoT | input合成/真实 | hidden合成/真实 | joint合成/真实 |','|---|---|---:|---:|---:|---:|']
for d,e in sorted({(r['dataset'],r['encoder']) for r in selected}):
 by={(r['space'],r['origin']):r for r in selected if (r['dataset'],r['encoder'])==(d,e)}
 pair=lambda space:' / '.join(f"{100*by[space,o]['final_macro_accuracy']:.3f}" for o in ('synthetic','real_diagnostic'))
 old=next(iter(by.values()))['original_tmot'];lines.append(f"| {d} | {e} | {100*old:.3f} | {pair('input')} | {pair('hidden')} | {pair('joint')} |")
lines+=['','## 选中的alpha','', '| 数据集 | 表示 | input合成/真实 | hidden合成/真实 | joint合成/真实 |','|---|---|---:|---:|---:|']
for d,e in sorted({(r['dataset'],r['encoder']) for r in selected}):
 by={(r['space'],r['origin']):r for r in selected if (r['dataset'],r['encoder'])==(d,e)}
 pair=lambda sp:' / '.join(str(by[sp,o]['alpha']) for o in ('synthetic','real_diagnostic'))
 lines.append(f"| {d} | {e} | {pair('input')} | {pair('hidden')} | {pair('joint')} |")
lines+=['','## 对方法改进的含义','', 'PhonemeSpectra+Mantis中，真实joint读出37.104%比真实input（FeCAM）35.596%高1.508个百分点；LSST+summary真实joint39.979%比真实input38.837%高1.142个百分点。这说明在这两个开发条件下，学习表示包含可利用的信息，但不代表普遍有效。','', '相同冻结函数下，PhonemeSpectra+Mantis的joint由真实统计37.104%降至合成统计35.596%；LSST+summary由39.979%降至37.607%。因此这些场景存在代表点统计近似造成的读出差距。但合成统计有时反而更好（如Handwriting+summary），不能把真实统计称作性能上界。','', 'Handwriting+Mantis中，无论合成还是真实统计，hidden/joint均只有30.769%，低于原TMoT44.231%，也低于之前原始表示ridge51.923%。这里不是补齐统计即可解决，学习表示/读出组合本身不合适。','', '下一步应在完整任务流中检验保持原始表示统计的读出与对合成点归一化类统计的约束，分别对照未改动TMoT、FeCAM、累计ridge及ER10。不可按数据集事后选择最优头并作为一个统一算法报告。当前离线诊断不作为持续学习主表结果。']
lines+=['','## 核查与适用范围','', '36个input真实统计配置逐项复现v59 FeCAM预测；6个原TMoT检查点的原始预测复现v54。所有输入与检查点SHA核验；网络参数、归一化参数、合成点与RNG前后不变；加权原始二阶矩与独立中心化协方差互核。保存预测、类别距离、真实标签与样本ID，所有216个指标独立重算。','', '这是最终阶段的读出替换诊断，不提供阶段平均分或遗忘曲线。各条件均按VAL选参，不能将同一开发集上的最好条件当作未经选择的稳健增益。合成点数和统计量内存与ER真实样本数不同，不能据此宣称等内存优势。','', '整体研究仍覆盖类新增和回归；分类诊断不替代回归验证，也不缩小一般多维时间序列持续学习目标。完整稳健优势尚未证明。']
Path('docs/CLASS_DISTANCE_REPORT.md').write_text('\n'.join(lines)+'\n')
