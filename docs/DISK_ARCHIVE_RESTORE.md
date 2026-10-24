# 历史模型归档与恢复

2026-10-24 第二次清理使用 `scripts/compact_disk.py`，范围限定于 TMoT。

- v45 保留每个数据集、编码器、方法的最佳最终准确率、最佳阶段平均准确率和 ridge=0.1 控制检查点，删除其余 64 个已完成验证扫描检查点。全部预测与指标保留；删除模型需按配置与源码版本重新训练。
- v1—v30 的已完成运行、单链接 `final.pt` 进行 gzip level 1 无损归档。压缩收益不足 5% 的保持原状。v31 及之后的模型（除上述 v45 扫描）不参与归档，以保留近期诊断和后续实验依赖。
- 每个归档在替换原文件前校验解压后的 SHA256 与原始文件一致。全部非模型证据文件在前后进行哈希复核。
- 不修改其他 WiMoT/WiFi 项目、原始数据、处理后数据、特征、预训练权重、运行环境。

归档后的历史检查点不能直接按原路径加载。恢复一个模型：

```bash
cd /root/autodl-tmp/TMoT
.venv/bin/python -m scripts.restore_archived_checkpoint \
  /root/autodl-tmp/tmot-artifacts/runs/QUEUE/RUN/final.pt \
  /root/autodl-tmp/tmot-artifacts/disk_compaction_20261024.json
```

命令要求完整归档清单，校验原始 SHA256，并保留压缩备份。运行前需要容纳解压模型的空间。不要批量解压全部旧模型。

归档清单记录每个文件的路径、原始/压缩大小和 SHA256；删除清单同时记录配置和源码版本。所有负面、零增益和正面结果都保留，压缩不改变实验结果。旧审计若要求 `final.pt` 必须先恢复对应文件。

## 完成结果

实际释放 19.59 GiB；可用空间从 17.57 GiB 增加至 37.16 GiB（df 显示 38G），使用率 89% → 76%。v45 删除 64 个未入选检查点，保留 44 个；历史文件处理统计为 {'archived_verified': 1539}。38,775 个非模型证据文件前后 SHA256 完全一致。恢复工具通过合成文件往返测试。完整清单及逐项日志位于 results/disk_compaction_20261024*。
