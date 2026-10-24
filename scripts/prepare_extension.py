"""Fixed three-dataset expansion; preserve official tests and existing results."""
import argparse,json
from pathlib import Path
from scripts.prepare_round3 import classification
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
inventory=json.loads((a.root/'extension_train_inventory.json').read_text());assert len(inventory)==3 and all(r['finite'] and not r['test_read'] for r in inventory)
for name,archive in (('phonemespectra','PhonemeSpectra'),('lsst','LSST'),('handwriting','Handwriting')):
    if any((a.root/'prepared'/f'{name}_{suffix}').exists() for suffix in ('manifest.json','train.npz','val.npz','test.npz')):raise FileExistsError(name)
    classification(a.root/'raw/extension',a.root/'prepared',name,archive)
