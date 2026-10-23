import numpy as np
from tmot.temporal_joint import temporal_descriptor,descriptor_dimension,JointTemporalBasis
from tmot.stable_residual import StableSemanticBasis


def test_temporal_descriptor_is_sample_local_and_retains_channel_relations():
    t=np.linspace(0,4*np.pi,32);s=np.sin(t);x=np.array([[s,s],[s,-s]])
    a=temporal_descriptor(x);assert a.shape==(2,descriptor_dimension(2))
    np.testing.assert_array_equal(a[:1],temporal_descriptor(x[:1]))
    np.testing.assert_allclose(a[:,64],[1,-1],atol=1e-6)
    assert np.isfinite(temporal_descriptor(np.zeros((3,4,2)))).all()


def test_zero_joint_fusion_exactly_matches_original_basis():
    rng=np.random.default_rng(8);original=rng.normal(size=(20,26)).astype('float32');joint=rng.normal(size=(20,17)).astype('float32')
    x=np.concatenate([original[:,:-6],joint,original[:,-6:]],axis=1)
    base=StableSemanticBasis(6,32).fit(original[:10]);fusion=JointTemporalBasis(6,17,32,strength=0).fit(x[:10])
    np.testing.assert_array_equal(base.transform(original),fusion.transform(x))
    model=JointTemporalBasis(6,17,32).fit(x[:10]);before=model.transform(x[:10]);model.transform(x[10:])
    np.testing.assert_array_equal(before,model.transform(x[:10]));assert np.isfinite(before).all()
