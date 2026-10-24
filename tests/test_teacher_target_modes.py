import numpy as np
from tmot.statistical_teacher import distance_probabilities,teacher_targets

def test_control_argmax_and_sharpening():
 d=np.array([[3.,2.,1.],[0.,0.,1.],[0.,0.,0.]])
 np.testing.assert_array_equal(teacher_targets(d),distance_probabilities(d))
 for mode in ('soft','sharp','hard'):
  p=teacher_targets(d,mode);np.testing.assert_allclose(p.sum(1),1);np.testing.assert_array_equal(p.argmax(1),d.argmin(1));assert np.isfinite(p).all()
 assert teacher_targets(d,'sharp')[0].max()>teacher_targets(d,'soft')[0].max()
 np.testing.assert_array_equal(teacher_targets(d,'hard')[1],[1,0,0])
 np.testing.assert_allclose(teacher_targets(d,'sharp'),teacher_targets(100*d,'sharp'))
