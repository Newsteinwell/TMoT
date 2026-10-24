"""Causal input calibration for adaptive functions; fixed base remains unchanged."""
import numpy as np
from .regularized_innovation import RegularizedInnovationLearner

class ArrivedCalibrationLearner(RegularizedInnovationLearner):
    def __init__(self,*args,calibration='first',**kwargs):
        if calibration not in ('first','arrived'):raise ValueError(calibration)
        super().__init__(*args,**kwargs)
        self.calibration=calibration;self.calibration_n=0;self.calibration_mean=None;self.calibration_m2=None
    def observe(self,x,y):
        x=np.asarray(x,np.float32)
        if self.calibration=='arrived':
            # Chan/Welford merge uses only newly arrived true input, never generated replay.
            z=x.astype(np.float64);n=len(z);mean=z.mean(0);m2=((z-mean)**2).sum(0)
            if self.calibration_n:
                total=self.calibration_n+n;delta=mean-self.calibration_mean
                self.calibration_m2+=m2+delta**2*(self.calibration_n*n/total)
                self.calibration_mean+=delta*(n/total)
            else:self.calibration_mean=mean;self.calibration_m2=m2
            self.calibration_n+=n
            # First stage follows existing float32 calibration exactly in both arms.
            if self.seen:
                sd=np.sqrt(np.maximum(self.calibration_m2/self.calibration_n,0))
                self.input_stats=(self.calibration_mean.astype(np.float32),np.maximum(sd,max(float(np.median(sd))*.01,1e-4)).astype(np.float32))
        super().observe(x,y)
    @property
    def persistent_bytes(self):
        extra=0 if self.calibration_mean is None else self.calibration_mean.nbytes+self.calibration_m2.nbytes+8
        return super().persistent_bytes+extra
