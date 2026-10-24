import json
from pathlib import Path
runs=json.loads(Path('results/statistical_teacher_v74_audit.json').read_text())['runs'];old=json.loads(Path('results/sequential_head_v73_audit.json').read_text())['runs'];selected=[]
lines=['# v74 全类别统计教师约束：机制验证','','训练源码8f6b722332747ceeb941e26d093d9148dad20013。6个开发数据集×2表示×4强度，48条完整轨迹；7项单元测试通过。没有TEST、没有新增盲测数据。超参数固定见协议，表中仅按最终VAL选择0.1/1/10的开发最佳强度。','','独立审计：所有lambda0轨迹精确复现v73 FT；所有分类教师预测精确复现v59 alpha10；独立重算协方差、回归正规方程与当前TRAIN教师目标；核验全部阶段指标、样本ID、输入哈希和最终学生检查点预测。','','分类为宏准确率百分数（↑），回归为RMSE（↓）。阶段平均对应同一最终选定配置。','','| 数据 | 表示 | FT最终 | 旧模型蒸馏最终 | 统计教师最终 | 学生最终 | 学生阶段平均 | 强度 |','|---|---|---:|---:|---:|---:|---:|---:|']
for d,e in sorted(set((r['dataset'],r['encoder']) for r in runs)):
 group=[r for r in runs if (r['dataset'],r['encoder'])==(d,e)];cl=group[0]['task']=='classification';key=lambda r:-r['final'] if cl else r['final'];best=min([r for r in group if r['strength']>0],key=key);ft=next(r for r in group if r['strength']==0);prev=min([r for r in old if r['dataset']==d and r['encoder']==e and r['strength']>0],key=key);scale=100 if cl else 1
 selected.append(dict(**best,ft_final=ft['final'],previous_distillation_final=prev['final']))
 lines.append(f"| {d} | {e} | {ft['final']*scale:.3f} | {prev['final']*scale:.3f} | {best['teacher_final']*scale:.3f} | {best['final']*scale:.3f} | {best['average']*scale:.3f} | {best['strength']:g} |")
lines+=['','## 最终阶段分类行为','','| 数据 | 表示 | 旧类宏准确率% | 旧样本判为最新类% |','|---|---|---:|---:|']
for r in selected:
 if r['task']=='classification':lines.append(f"| {r['dataset']} | {r['encoder']} | {100*r['old_macro']:.3f} | {100*r['old_predicted_as_new']:.3f} |")
lines+=['','## 资源与比较边界','','无历史原始样本缓存，但训练与推断前持有统计教师；教师内存逐阶段记录于summary，不能将学生检查点尺寸当作全算法内存。只保留学生最终模型不表示教师统计没有成本。当前TRAIN生成的教师目标存盘仅为审计，不供后续任务训练。','','本轮教师统一alpha10/ridge0.1，未为每个数据重新选择，不能把它等同于各数据选参后的最强FeCAM/ridge。FT与蒸馏使用同样的当前数据优化步数，统计教师计算有额外成本；与ER的历史重采样训练不属于等计算预算消融。','','归档v41/v48老检查点释放521818112字节（约0.49GiB），4216个非模型证据文件hash保持不变，所有压缩模型均解压hash校验后才移除原副本。详见results/archive_before_v74.json。']
lines+=['','## 结论与下一步','','全类别教师并未解决类新增遗忘。最终VAL选择的六个分类配置中，旧样本被判为最新类的比例仍为95.94%–100%；学生全部显著弱于对应统计教师。即使损失能感知新旧类总概率偏移，仅在当前任务输入上约束学生仍不足以保留旧类决策区域。该实验没有单独隔离蒸馏温度、软目标质量和网络表达能力，不能把失败全部归因于某一个因素。','','回归学生在Beijing+summary和两个Energy配置优于固定ridge教师，在两个Air和Beijing+Mantis弱于教师；其中Energy为旧开发集，不能计作新增泛化证据。多个最佳强度落在网格上限10，不声称充分优化；也不因为存在局部改善而扩大网格追逐开发VAL。总体未满足多数数据集明确优势。','','此模块不并入默认TMoT。下一步应检验历史输入覆盖与稳定读出的直接保留：在同一教师、同一网络、同一更新步数下，将当前输入蒸馏与历史统计生成输入上的约束作对照，并同时保留独立教师读出作为参照。必要时采用预算内真实样本作为诊断，仍遵守每个历史任务不超过10%。这能区分历史支撑缺失与合成分布误差，避免继续只调输出约束强度。']
Path('results/statistical_teacher_v74_selected.json').write_text(json.dumps(selected,indent=2));Path('docs/STATISTICAL_TEACHER_REPORT.md').write_text('\n'.join(lines)+'\n')
for r in selected:print(r['dataset'],r['encoder'],'student',r['final'],'teacher',r['teacher_final'],'strength',r['strength'],'old_as_new',r.get('old_predicted_as_new'))
