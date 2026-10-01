import numpy as np, sys
from kin import fk_arm, ik
from approach import HOME, DOWN_R
for reach in [0.52,0.56,0.58,0.60]:
    for z in [0.623,0.75]:
        q,_=ik(np.array(HOME,float),np.array([reach,0,z-0.3947]),DOWN_R,tool=0.0,iters=300)
        p,R,_,_=fk_arm(q,0.0)
        print(reach,z,np.round(p-[reach,0,z-0.3947],4),np.round(np.linalg.norm(R-DOWN_R),4),np.round(q,2))
