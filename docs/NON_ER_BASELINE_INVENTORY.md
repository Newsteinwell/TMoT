# 已有非ER基线索引

本表是历史结果索引，不是统一协议排行榜。不同版本表示、数据划分和调参预算可能不同；不得混选后直接宣称方法优势。完整配置、输入hash和来源保存在 results/non_er_baseline_inventory.json。

| 方法 | 历史记录数 | 数据集覆盖 |
|---|---:|---|
| fecam_adapted | 132 | articulary, dailysports, epilepsy, handwriting, har, lsst, natops, pendigits, phonemespectra, racketsports, uwave |
| ft | 16 | dailysports, energy, har, uwave |
| ranpac | 12 | dailysports, har, uwave |
| ridge | 1002 | air_quality, articulary, beijing_pm25, dailysports, energy, epilepsy, handwriting, har, lsst, natops, pendigits, phonemespectra, racketsports, uwave |
| slda | 55 | articulary, dailysports, epilepsy, har, natops, pendigits, racketsports, uwave |
| stable_residual | 128 | air_quality, beijing_pm25, dailysports, energy, har, uwave |

ranpac 是早期随机特征+累计ridge适配，不能当作完整原论文复现；fecam_adapted 是依据固定来源代码实现的时序特征适配，来源与许可证见 third_party/fecam。SLDA与FeCAM仅分类。当前较难分类集尚缺SLDA/RanPAC等统一表示的结果；EWC、LwF、DER++、A-GEM尚未找到已实现实验，不能填写结果。

本轮FeCAM v59补充PhonemeSpectra/LSST/Handwriting，ER跨任务网络 v58 补充暖启动与监督头；后续仍须补充正则化/蒸馏/其他重放基线，不能把这两轮当作完整基线矩阵。
