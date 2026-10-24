# v82：风险预算与单步表示更新验证


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

已完成输入组：10/10。仅TRAIN/VAL，seed42/order17；预算和表示维数固定，不按VAL挑赢家。

分类变化为相对冻结投影的准确率百分点，回归变化为相对冻结投影的RMSE下降百分比。不同表示不是独立数据集。

|数据/表示|方法|最终|阶段平均|相对冻结变化|接受/可更新阶段|
|---|---|---:|---:|---:|---:|
|air_quality/mantis|frozen|1.08740|0.79636|+0.0000|0/4|
|air_quality/mantis|unguarded|1.02446|0.79077|+5.7885|4/4|
|air_quality/mantis|center_e001|1.02446|0.79077|+5.7885|4/4|
|air_quality/mantis|center_e01|1.02446|0.79077|+5.7885|4/4|
|air_quality/mantis|center_e1|1.02446|0.79077|+5.7885|4/4|
|air_quality/mantis|spectral_e01|1.02446|0.79077|+5.7885|4/4|
|air_quality/mantis|full_ridge|1.01624|0.83644|+6.5445|0/0|
|air_quality/summary|frozen|0.91567|0.79968|+0.0000|0/4|
|air_quality/summary|unguarded|1.02236|0.86552|-11.6519|4/4|
|air_quality/summary|center_e001|0.99018|0.83755|-8.1372|4/4|
|air_quality/summary|center_e01|0.99018|0.83755|-8.1372|4/4|
|air_quality/summary|center_e1|1.02236|0.86552|-11.6519|4/4|
|air_quality/summary|spectral_e01|0.99018|0.83755|-8.1372|4/4|
|air_quality/summary|full_ridge|0.93401|0.89749|-2.0030|0/0|
|beijing_pm25/mantis|frozen|65.44528|72.88148|+0.0000|0/4|
|beijing_pm25/mantis|unguarded|49.27703|63.93981|+24.7050|4/4|
|beijing_pm25/mantis|center_e001|50.06612|64.09763|+23.4993|4/4|
|beijing_pm25/mantis|center_e01|50.06612|64.09763|+23.4993|4/4|
|beijing_pm25/mantis|center_e1|49.27703|63.93981|+24.7050|4/4|
|beijing_pm25/mantis|spectral_e01|50.06612|64.09763|+23.4993|4/4|
|beijing_pm25/mantis|full_ridge|44.70767|52.94773|+31.6869|0/0|
|beijing_pm25/summary|frozen|72.55910|79.94022|+0.0000|0/4|
|beijing_pm25/summary|unguarded|58.25776|71.83551|+19.7099|4/4|
|beijing_pm25/summary|center_e001|58.25776|71.83551|+19.7099|4/4|
|beijing_pm25/summary|center_e01|58.25776|71.83551|+19.7099|4/4|
|beijing_pm25/summary|center_e1|58.25776|71.83551|+19.7099|4/4|
|beijing_pm25/summary|spectral_e01|58.25776|71.83551|+19.7099|4/4|
|beijing_pm25/summary|full_ridge|43.64985|53.40077|+39.8423|0/0|
|handwriting/mantis|frozen|21.15385|23.46268|+0.0000|0/12|
|handwriting/mantis|unguarded|26.92308|32.50411|+5.7692|11/12|
|handwriting/mantis|center_e001|21.15385|23.46268|+0.0000|0/12|
|handwriting/mantis|center_e01|21.15385|23.46268|+0.0000|0/12|
|handwriting/mantis|center_e1|21.15385|23.46268|+0.0000|8/12|
|handwriting/mantis|spectral_e01|23.07692|23.85480|+1.9231|11/12|
|handwriting/mantis|full_ridge|26.92308|23.58567|+5.7692|0/0|
|handwriting/mantis|full_fecam1|36.53846|30.96194|+15.3846|0/0|
|handwriting/summary|frozen|19.23077|19.72604|+0.0000|0/12|
|handwriting/summary|unguarded|21.15385|15.39900|+1.9231|11/12|
|handwriting/summary|center_e001|19.23077|19.72604|+0.0000|0/12|
|handwriting/summary|center_e01|19.23077|19.72604|+0.0000|0/12|
|handwriting/summary|center_e1|19.23077|19.72604|+0.0000|8/12|
|handwriting/summary|spectral_e01|19.23077|18.68389|+0.0000|11/12|
|handwriting/summary|full_ridge|11.53846|11.64196|-7.6923|0/0|
|handwriting/summary|full_fecam1|28.84615|22.31585|+9.6154|0/0|
|lsst/mantis|frozen|49.04207|59.39591|+0.0000|0/6|
|lsst/mantis|unguarded|52.98391|61.13137|+3.9418|6/6|
|lsst/mantis|center_e001|49.04207|59.39591|+0.0000|0/6|
|lsst/mantis|center_e01|49.04207|59.39591|+0.0000|3/6|
|lsst/mantis|center_e1|49.04207|59.42471|+0.0000|6/6|
|lsst/mantis|spectral_e01|49.19558|59.75557|+0.1535|6/6|
|lsst/mantis|full_ridge|49.68147|66.60053|+0.6394|0/0|
|lsst/mantis|full_fecam1|43.03936|48.25552|-6.0027|0/0|
|lsst/summary|frozen|30.20521|47.03500|+0.0000|0/6|
|lsst/summary|unguarded|33.80249|49.24660|+3.5973|6/6|
|lsst/summary|center_e001|30.20521|47.03500|+0.0000|0/6|
|lsst/summary|center_e01|30.20521|47.03500|+0.0000|2/6|
|lsst/summary|center_e1|30.25129|47.06463|+0.0461|6/6|
|lsst/summary|spectral_e01|31.42064|47.35122|+1.2154|6/6|
|lsst/summary|full_ridge|23.59456|43.72483|-6.6106|0/0|
|lsst/summary|full_fecam1|20.11793|31.83802|-10.0873|0/0|
|phonemespectra/mantis|frozen|28.65762|44.85641|+0.0000|0/18|
|phonemespectra/mantis|unguarded|28.80845|44.28971|+0.1508|18/18|
|phonemespectra/mantis|center_e001|28.65762|44.85641|+0.0000|0/18|
|phonemespectra/mantis|center_e01|28.65762|44.85641|+0.0000|0/18|
|phonemespectra/mantis|center_e1|28.65762|44.89776|+0.0000|18/18|
|phonemespectra/mantis|spectral_e01|29.41176|44.92348|+0.7541|18/18|
|phonemespectra/mantis|full_ridge|20.51282|34.69016|-8.1448|0/0|
|phonemespectra/mantis|full_fecam1|29.86425|43.07686|+1.2066|0/0|
|phonemespectra/summary|frozen|9.04977|18.20790|+0.0000|0/18|
|phonemespectra/summary|unguarded|9.04977|17.95973|+0.0000|18/18|
|phonemespectra/summary|center_e001|9.04977|18.20790|+0.0000|0/18|
|phonemespectra/summary|center_e01|9.04977|18.20790|+0.0000|0/18|
|phonemespectra/summary|center_e1|9.04977|18.20790|+0.0000|0/18|
|phonemespectra/summary|spectral_e01|8.74811|17.92342|-0.3017|18/18|
|phonemespectra/summary|full_ridge|6.63650|15.65442|-2.4133|0/0|
|phonemespectra/summary|full_fecam1|8.74811|15.71174|-0.3017|0/0|

## 理论核验与解释边界

所有已完成组的候选均在选择结束后独立核对真实历史风险；分类逐类检查正分数增量不超过中心/谱证书，回归核对解析风险与显式误差。最终预测从反序列化状态重现。这些是浮点容差内的数值检查，不代替无限精度证明。

无约束版本仍要求当前收益门槛，未接受更新不等于风险预算拒绝。分类当前CE下降不保证VAL准确率提高。预算只控制新增类别、统计重拟合之后表示适配的额外影响。谱回归与中心回归是同一精确风险机制，不计为两种独立创新。

完整512维ridge/FeCAM为固定参数对照，不是穷尽调参后的最强基线；Beijing未使用历史目标残差先验。多数数据集整体优势需要另行完整协议验证，本轮不据此宣布达标。

## 更新与拒绝原因（候选点相关，不作独立统计样本）

|数据/表示|方法|有用收益不足的阶段|证书松弛拒绝点|正增量保守性拒绝点|真实风险冲突拒绝点|
|---|---|---:|---:|---:|---:|
|air_quality/mantis|center_e001|0/4|0|0|0|
|air_quality/mantis|center_e01|0/4|0|0|0|
|air_quality/mantis|center_e1|0/4|0|0|0|
|air_quality/mantis|spectral_e01|0/4|0|0|0|
|air_quality/summary|center_e001|0/4|0|0|1|
|air_quality/summary|center_e01|0/4|0|0|1|
|air_quality/summary|center_e1|0/4|0|0|0|
|air_quality/summary|spectral_e01|0/4|0|0|1|
|beijing_pm25/mantis|center_e001|0/4|0|0|1|
|beijing_pm25/mantis|center_e01|0/4|0|0|1|
|beijing_pm25/mantis|center_e1|0/4|0|0|0|
|beijing_pm25/mantis|spectral_e01|0/4|0|0|1|
|beijing_pm25/summary|center_e001|0/4|0|0|0|
|beijing_pm25/summary|center_e01|0/4|0|0|0|
|beijing_pm25/summary|center_e1|0/4|0|0|0|
|beijing_pm25/summary|spectral_e01|0/4|0|0|0|
|handwriting/mantis|center_e001|1/12|46|80|6|
|handwriting/mantis|center_e01|1/12|78|54|0|
|handwriting/mantis|center_e1|1/12|93|24|0|
|handwriting/mantis|spectral_e01|1/12|5|56|0|
|handwriting/summary|center_e001|1/12|40|90|2|
|handwriting/summary|center_e01|1/12|75|57|0|
|handwriting/summary|center_e1|1/12|86|26|0|
|handwriting/summary|spectral_e01|1/12|3|54|0|
|lsst/mantis|center_e001|0/6|27|22|20|
|lsst/mantis|center_e01|0/6|43|16|6|
|lsst/mantis|center_e1|0/6|45|2|0|
|lsst/mantis|spectral_e01|0/6|9|16|6|
|lsst/summary|center_e001|0/6|22|29|16|
|lsst/summary|center_e01|0/6|40|18|7|
|lsst/summary|center_e1|0/6|39|4|0|
|lsst/summary|spectral_e01|0/6|7|18|7|
|phonemespectra/mantis|center_e001|0/18|78|99|21|
|phonemespectra/mantis|center_e01|0/18|143|54|1|
|phonemespectra/mantis|center_e1|0/18|148|0|0|
|phonemespectra/mantis|spectral_e01|0/18|36|54|1|
|phonemespectra/summary|center_e001|0/18|77|118|0|
|phonemespectra/summary|center_e01|0/18|159|36|0|
|phonemespectra/summary|center_e1|0/18|195|0|0|
|phonemespectra/summary|spectral_e01|0/18|62|34|0|

拒绝归因只基于已到达TRAIN的事后审计，不反馈给学习器；没有收益的候选不进入风险拒绝计数。实际风险不超预算但正增量超预算，说明禁止正负抵消的保守性；正增量也不超预算而证书拒绝，说明可计算上界的松弛。这里的“实际风险”仍是历史经验损失，不是VAL/TEST泛化风险。
