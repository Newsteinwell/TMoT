import numpy as np
from tmot.kernel_transport import KernelBasis,KernelTransportLearner
from tmot.moments import augment

def test_online_transport_matches_explicit_all_seen_ridge():
    rng=np.random.default_rng(42);x=rng.normal(size=(30,10)).astype('float32');y=rng.normal(size=(30,2))
    basis=KernelBasis(2,dimension=8).fit(x[:10]);phi=basis.transform(x)
    model=KernelTransportLearner(24,task='regression',ridge=.1)
    for ids in np.array_split(np.arange(30),3):model.observe(phi[ids],y[ids])
    z=augment(phi)@model.h.T;penalty=np.eye(25)*.1;penalty[-1,-1]=0
    reference=np.linalg.solve(z.T@z/30+penalty,z.T@y/30)
    np.testing.assert_allclose(model.predict(phi),z@reference,atol=1e-8)
