# 第二轮执行状态

目标仍在进行，未完成最终测试审计及报告。

- diagnosis_v2：112条完成，全部预测指标复算一致；14对head/ridge比较的adapter、head、基底完全一致。
- transport_v3：52条完成，无失败；完整逐次汇总已提交。HAR/DailySports改善，UWave弱项保留。
- followups_v4：12条完成。HAR真实DIL路径可运行但full未超过frozen/no_hist；UWave秩补全val提升，尚待test。
- locked_test_v5：34条锁定测试，控制器PID8210（后续须实际核对存活）；代码ceb8556，summary源标识使用别名locked_test_v5，见SOURCE_SNAPSHOTS.md。
- locked_extension_v6：30条锁定追加测试，第二类序与UWave秩补全；训练代码93ef479。
- 用户要求继续提速，调度上限改为16总并发。动态统计其他控制器/孤立子进程，按剩余槽位补充任务；现有训练不杀、不覆盖。
- 初始扩展控制器8656改为动态槽位调度；新PID查服务器locked_extension_v6_dynamic.log对应进程。已经存在的运行目录跳过；审计以最终summary为准。
- 最新检查13项测试通过；磁盘剩余约109GB。用户已授权清理无价值中间产物，当前无清理需要。
- 下步：核验两个测试队列终态、逐预测审计和源代码哈希；汇总同基底基线、两类序结果、回归基线及限制；同步服务器/GitHub并产出最终报告。不能根据test继续挑参。
