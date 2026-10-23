import importlib.util
from pathlib import Path
import os
import pytest
spec=importlib.util.spec_from_file_location('checkpoint_dedup',Path(__file__).parents[1]/'scripts/deduplicate_checkpoints.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_exact_dedup_keeps_paths_and_refuses_changed_contents(tmp_path):
    queue=tmp_path/'runs/q';queue.mkdir(parents=True);(queue/'complete.json').write_text('[]')
    paths=[]
    for run in ('a','b'):
        folder=queue/run;folder.mkdir();(folder/'summary.json').write_text('{}');p=folder/'final.pt';p.write_bytes(b'checkpoint');paths.append(p)
    report={'duplicate_groups':[{'size':10,'sha256':module.digest(paths[0]),'files':[{'path':str(p)} for p in paths]}]}
    paths[1].write_bytes(b'different!')
    with pytest.raises(ValueError):module.apply(report,tmp_path)
    assert not os.path.samefile(*paths)
    paths[1].write_bytes(b'checkpoint');result=module.apply(report,tmp_path)
    assert os.path.samefile(*paths) and all(p.read_bytes()==b'checkpoint' for p in paths)
    assert result['reclaimed_bytes']==10
    assert module.apply(report,tmp_path)['changed']==[]
