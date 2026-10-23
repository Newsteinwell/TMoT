"""Numerical invariants for the migrated density/OSCL/calibration core."""
import numpy as np
from tmot.wimot_core.statistics import moments
from tmot.wimot_core.readout import head_from,StudentDecoder,TargetRiskPair,cases
from tmot.wimot_core.calibration import subtract

def test_density_and_pair_optimization():
    rng=np.random.default_rng(12);y=np.repeat(np.arange(3),16)
    x=rng.normal(size=(48,4))+np.eye(4)[y]*3
    stats=moments(x,y,3)
    pair=TargetRiskPair([stats],(0,1),0,maxiter=20)
    assert pair.log_objective(pair.labels)<=pair.status['fixed_log_objective']+1e-10
    base=head_from([stats],3)
    for df,gate in cases():
        p=StudentDecoder(base,df,gate).predict_proba(x)
        assert np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
        assert (p.argmax(1)==y).mean()>.8

def test_fold_subtraction_matches_direct_moments():
    rng=np.random.default_rng(23);y=np.repeat(np.arange(3),12);x=rng.normal(size=(36,5))
    held=np.arange(0,len(y),3);keep=np.ones(len(y),bool);keep[held]=False
    removed=subtract([moments(x,y,3)],x[held],y[held],np.zeros(len(held),int))[0]
    direct=moments(x[keep],y[keep],3)
    np.testing.assert_allclose(removed.means,direct.means,atol=1e-12)
    np.testing.assert_allclose(removed.covariances,direct.covariances,atol=1e-12)
