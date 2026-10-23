# Data and attribution

Raw datasets must remain in the external artifact directory. This repository distributes no data.

- HAR: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
  Download: https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
  Official subject-disjoint train/test; largest four training subject IDs held for validation. Nine inertial channels, original windows. Credit Anguita et al., A Public Domain Dataset for Human Activity Recognition Using Smartphones (2013).
- UWaveGestureLibrary: https://www.timeseriesclassification.com/description.php?Dataset=UWaveGestureLibrary
  Download: https://www.timeseriesclassification.com/aeon-toolkit/UWaveGestureLibrary.zip
  Downloaded archive contains 120 training and 320 test cases (3 channels, length315), different from the website table of2238/2241; SHA256 and actual shapes in the manifest are authoritative. This small archive variant is explicitly a pilot, not a replication of the larger version. Official train/test; stratified20% of train for validation. Subject IDs absent: no claim of subject-disjoint validation. Credit Liu et al., uWave (2009), UEA archive maintainers.
- Daily and Sports Activities: https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities
  Download: https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip
  Original19 classes, 45channels, 125samples; p1–p5 train, p6 validation, p7–p8 test. Credit Altun, Barshan, Tuncel (2010). This differs from TSCIL's processed18-class variant.
- Appliances Energy: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
  Download: https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip
  Credit Candanedo, Feldheim, Deramaix (2017), DOI10.24432/C5VC8G, UCI CC BY4.0. Temporal windows and target definition are specified in PROTOCOL.md.

MantisV2: https://github.com/vfeofanov/mantis and https://huggingface.co/paris-noah/MantisV2 (Apache2.0 model card). No pretrained weights distributed here. Model file hashes accompany feature caches.

Related foundations: WiMoT https://github.com/Newsteinwell/WiMoT (MIT; source commit7d3ce9512198867ef827721cbeb4fde53c16424d); OSCL https://openreview.net/forum?id=Vf6WcUDnY7c; TSCIL https://github.com/zqiao11/TSCIL; ACIL https://proceedings.neurips.cc/paper/2022/hash/4b74a42fc81fc7ee252f6bcb6e26c8be-Abstract.html; RanPAC https://arxiv.org/abs/2307.02251.
