# 第二轮执行状态

目标仍在进行：完整迁移与验证WiMoT关键模块，检验瓶颈和读出，覆盖类新增及回归。

- diagnosis_v2：112条全部生成summary；验证指标已同步GitHub，见DIAGNOSIS_V2.md。逐预测及同状态读出审计待完成。
- 旧2并发控制器5626被终止调度，训练子进程保留；8并发接续控制器6565已完成，没有重复覆盖实验。
- transport_v3：训练版本ae4055a；52条验证轨迹，8并发；队列PID7177。
- 路径：`/root/autodl-tmp/tmot-artifacts/runs/transport_v3`；日志`logs/transport_v3.log`。
- 运行：`.venv/bin/python scripts/run_transport_queue.py /root/autodl-tmp/tmot-artifacts --workers 8`，环境TMOT_SOURCE_COMMIT=ae4055a。
- 12项测试全部通过（2.92秒）：矩风险、类新增/回归、双分支冻结及PCA秩、OSCL与校准。
- 下步：观察真实运行；完成预测和检查点一致性审计；分析模块消融；固定配置测试集验证，并根据val检查回归正则化。
- 用户已授权空间不足时清理非关键中间产物；最近可用117GB，无需清理。
- 完整目标尚未完成，不把合成测试通过当作实际数据实验结论。
