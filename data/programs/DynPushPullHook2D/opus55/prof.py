import time, numpy as np
from approach import GeneratedApproach, rect_poly
class Sp: 
    low=np.array([-0.05,-0.05,-0.0654,-0.1,-0.02]); high=-low
ap=GeneratedApproach(Sp(),None,{})
ap.hw=0.107; ap.l1=1.4; ap.l2=0.583; ap.L0=np.array([1.72,0.045]); ap.arm0=0.24; ap.dphi=0.0
obs=[rect_poly(1.5,2.2,0.3,0.4,0.4), rect_poly(1.0,2.5,0.3,0.4,0.4), rect_poly(2.0,2.0,0.3,0.4,0.4)]
t=time.time()
for i in range(10): r=ap.path_free((0.5,0.3,0.0,0.24),(1.0,1.4,1.57,0.24),obs,True)
print(r,(time.time()-t)/10)
