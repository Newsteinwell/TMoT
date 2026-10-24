"""Summarize fixed v82 outputs without selecting configurations by VAL."""
import argparse,json
from pathlib import Path
import statistics as stats
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
assert len(list(a.root.glob("*/summary.json"))) == 10, "v82 report requires all 10 completed groups"
rows=[];details=[];diagnostics=[]
for path in sorted(a.root.glob('*/summary.json')):
 s=json.loads(path.read_text());audit=json.loads((path.parent/'audit.json').read_text());dataset=s['config']['dataset'];encoder=s['config']['encoder'];cl=s['task']=='classification'
 assert s['all_proposals_verified'] and s['final_reload_exact']
 for arm,metrics in s['results'].items():
  steps=[r for r in audit if r['arm']==arm and r['stage']>1];props=[q for r in steps for q in r['proposals']];acc=[q for q in props if q['accepted']]
  base=s['results']['frozen'];delta=(metrics['final']-base['final'])*100 if cl else 100*(base['final']-metrics['final'])/base['final']
  row=dict(dataset=dataset,encoder=encoder,task=s['task'],arm=arm,**metrics,delta_frozen=delta,updates_possible=len(steps),proposals=len(props),max_positive_violation=max([v['empirical_positive_gap']-v['certificate']['bound'] for v in props],default=0.),mean_alpha=float(stats.mean([q['alpha'] for q in acc])) if acc else None,mean_gain=float(stats.mean([q['gain'] for q in acc])) if acc else None,mean_bound=float(stats.mean([q['certificate']['bound'] for q in acc])) if acc else None)
  rows.append(row)
  if props:
   ratios=[q['certificate']['bound']/q['empirical_positive_gap'] for q in props if q['empirical_positive_gap']>1e-10]
   diagnostics.append(dict(dataset=dataset,encoder=encoder,arm=arm,candidates=len(props),median_bound_to_positive_gap=stats.median(ratios) if ratios else None,zero_positive_gap=sum(q['empirical_positive_gap']<=1e-10 for q in props),max_slack=max(q['certificate_slack'] for q in props)))
  for r in steps:
   pp=r['proposals'];minimum=r.get('minimum_gain',0);armijo=r.get('directional_descent',0)
   eligible=[q for q in pp if q['gain']>=max(minimum,1e-4*q['alpha']*armijo)]
   budget={'center_e001':.001,'center_e01':.01,'center_e1':.1,'spectral_e01':.01}.get(arm)
   reason_counts={}
   if budget is not None:
    for q in eligible:
     if q['accepted']:key='accepted'
     elif q['certificate']['bound']<=budget:key='other'
     elif q['empirical_positive_gap']<=budget:key='certificate_slack'
     elif q['actual_history_risk_delta']<=budget:key='positive_part_conservatism'
     else:key='actual_risk_conflict'
     reason_counts[key]=reason_counts.get(key,0)+1
   details.append(dict(rejection_candidates=reason_counts,dataset=dataset,encoder=encoder,arm=arm,stage=r['stage'],accepted=r['accepted'],eligible_count=len(eligible),reason=r['reason'],no_useful_gain=not bool(eligible)))
a.out.mkdir(parents=True,exist_ok=True)
(a.out/'certified_v82_summary.json').write_text(json.dumps(dict(groups=len(list(a.root.glob('*/summary.json'))),rows=rows,steps=details,diagnostics=diagnostics),indent=2))
lines=['# v82：风险预算与单步表示更新验证','',f'已完成输入组：{len(list(a.root.glob("*/summary.json")))}/10。仅TRAIN/VAL，seed42/order17；预算和表示维数固定，不按VAL挑赢家。','', '分类变化为相对冻结投影的准确率百分点，回归变化为相对冻结投影的RMSE下降百分比。不同表示不是独立数据集。','', '|数据/表示|方法|最终|阶段平均|相对冻结变化|接受/可更新阶段|','|---|---|---:|---:|---:|---:|']
for r in rows:
 factor=100 if r['task']=='classification' else 1
 lines.append(f"|{r['dataset']}/{r['encoder']}|{r['arm']}|{factor*r['final']:.5f}|{factor*r['average']:.5f}|{r['delta_frozen']:+.4f}|{r['accepted']}/{r['updates_possible']}|")
lines+=['','## 理论核验与解释边界','','所有已完成组的候选均在选择结束后独立核对真实历史风险；分类逐类检查正分数增量不超过中心/谱证书，回归核对解析风险与显式误差。最终预测从反序列化状态重现。这些是浮点容差内的数值检查，不代替无限精度证明。','','无约束版本仍要求当前收益门槛，未接受更新不等于风险预算拒绝。分类当前CE下降不保证VAL准确率提高。预算只控制新增类别、统计重拟合之后表示适配的额外影响。谱回归与中心回归是同一精确风险机制，不计为两种独立创新。','','完整512维ridge/FeCAM为固定参数对照，不是穷尽调参后的最强基线；Beijing未使用历史目标残差先验。多数数据集整体优势需要另行完整协议验证，本轮不据此宣布达标。']
lines+=['','## 更新与拒绝原因（候选点相关，不作独立统计样本）','','|数据/表示|方法|有用收益不足的阶段|证书松弛拒绝点|正增量保守性拒绝点|真实风险冲突拒绝点|','|---|---|---:|---:|---:|---:|']
for row in rows:
 if row['arm'] not in ('center_e001','center_e01','center_e1','spectral_e01'):continue
 subset=[v for v in details if (v['dataset'],v['encoder'],v['arm'])==(row['dataset'],row['encoder'],row['arm'])]
 totals={key:sum(v['rejection_candidates'].get(key,0) for v in subset) for key in ('certificate_slack','positive_part_conservatism','actual_risk_conflict')}
 lines.append(f"|{row['dataset']}/{row['encoder']}|{row['arm']}|{sum(v['no_useful_gain'] for v in subset)}/{len(subset)}|{totals['certificate_slack']}|{totals['positive_part_conservatism']}|{totals['actual_risk_conflict']}|")
lines+=['','拒绝归因只基于已到达TRAIN的事后审计，不反馈给学习器；没有收益的候选不进入风险拒绝计数。实际风险不超预算但正增量超预算，说明禁止正负抵消的保守性；正增量也不超预算而证书拒绝，说明可计算上界的松弛。这里的“实际风险”仍是历史经验损失，不是VAL/TEST泛化风险。']
conclusions = """
## 本轮结论与决策

本轮是固定的机制验证：5个数据集、2种表示、单种子和单类别顺序，共76条评估轨迹、4176个候选更新。全部候选通过独立历史TRAIN审计，未发现浮点容差外的证书覆盖或预算检查失败，最终预测均通过状态重载复现。实现的6项单元测试及分类/回归两阶段冒烟检查通过。数值结果支持实现一致性，不构成对任意数据分布的证明。

分类72次可更新机会中，无风险约束接受70次；中心界ε=0.001、0.01、0.1分别接受0、5、46次，谱界ε=0.01接受70次。尤其在谱界实际接受的70个同一候选上，中心证书都超过0.01，而谱证书不超过0.01：这是可计算上界松弛阻止有效更新的直接证据，并非仅比较两条分叉轨迹。其余两次即使无约束也没有满足当前训练收益门槛。

在中心界ε=0.01的分类候选中，538个有用候选被证书的额外松弛拒绝，235个因逐点正增量禁止抵消而被拒绝，14个实际历史经验风险增量超过预算。候选来自同一回溯路径，数量不能作为独立样本检验。谱界仍留下122个额外松弛拒绝点和232个正增量保守性拒绝点，但回溯找到可接受步长，因此本轮分类更新的主要瓶颈已不再是“完全不能动”。

有效更新不等于泛化收益。谱界相对冻结投影在6个分类条件中4升、1平、1降；Handwriting/Mantis提升1.923个百分点，LSST/summary提升1.215个百分点，PhonemeSpectra/summary下降0.302个百分点。它们是相对本轮冻结32维投影的收益，不能写成超过原有最强基线。Handwriting/Mantis的谱界最终准确率23.077%，固定完整维FeCAM为36.538%。本轮FeCAM固定alpha=1，也不能替代此前充分比较过的强基线。

回归的历史MSE增量可以用联合二阶统计量精确计算，但仍不保证VAL：Air Quality/summary在ε=0.01时RMSE从0.91567变为0.99018，恶化8.14%；Beijing/Mantis从65.445降至50.066，仍弱于本轮完整维ridge的44.708。投影和读出本身的性能损失，不能归咎于预算约束，也不能通过放宽预算自动解决。

建议保留：固定基底中的统计记忆、独立风险审计、谱界回溯、回归精确风险差、允许拒绝更新的机制。中心界适合作为便宜的提前通过检查，不宜单独承担最终拒绝；无需转向大规模SDP。暂不把固定32维共享投影作为最终统一架构，优先分析其丢失的信息和读出校准。当前只证明训练代理目标存在可行收益，没有证明准确率或未来任务收益；不能将验证集观察反推为光滑常数或理论保证。该轮不支持“稳定超过强基线”的结论。

## 复现与资源

训练源码提交：bbf115946c0a2de9785799daf50e64993c9700a0。协议在正式运行前固定，见CERTIFIED_UPDATE_PROTOCOL.md。任务使用6个CPU工作进程，各1条数值计算线程；矩阵规模小，本轮没有使用GPU。最慢单输入组约307秒，全部任务完成后停止，未自动发起下一轮。历史数据只在候选选择结束后用于审计，不回流决策；新增类别竞争与统计重拟合的影响不在表示更新预算内。

原始JSON证据压缩包：results/certified_v82_evidence.tar.gz；汇总：results/certified_v82_summary.json。证据包含全部10组summary/audit、manifest、complete和状态日志，不含原始数据或模型权重；完整模型及逐阶段预测保留于服务器/root/autodl-tmp/tmot-artifacts/runs/certified_v82。解压后可运行scripts/report_certified.py重新生成本报告。下一轮若进行，应预先固定对照和预算，不依据这轮VAL结果循环筛选并报告赢家。
"""
lines[2:2] = [conclusions]

(a.out/'CERTIFIED_UPDATE_REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:16]))
