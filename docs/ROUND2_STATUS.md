# 第二轮执行状态

目标仍在进行：完整迁移WiMoT关键模块，检验瓶颈和读出，覆盖类新增及回归。

- 诊断训练版本：d8735b5；服务器SOURCE_COMMIT在队列结束前保留该训练版本。
- 活跃队列PID：5626；启动路径 `/root/autodl-tmp/TMoT`。
- 命令：`.venv/bin/python scripts/run_diagnostics.py /root/autodl-tmp/tmot-artifacts`。
- 输出：`/root/autodl-tmp/tmot-artifacts/runs/diagnosis_v2`；日志 `logs/diagnosis_v2.log`。
- 112条验证轨迹，2 worker，每条4小时硬超时，已有目录不覆盖、不自动重跑。
- 本次进程检查：队列及两个训练子进程存活；HAR的64/128维及部分256维配置已完成，没有已报告失败。
- 新增数值核心：53c105b；服务器7项测试全部通过（2.24秒）。未修改正在运行的训练实现。
- 下一步：完成验证集诊断汇总及状态一致性核验；接入通用双分支/PCA、类新增稀疏读出与校准、回归共享模块；跑消融并发布结果。
- 完整模块迁移与下一轮结果尚未完成；本文件不是完成声明。
