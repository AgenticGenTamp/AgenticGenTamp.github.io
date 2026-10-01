import numpy as np
from kin import fk_arm, ik
from approach import HOME, DOWN_R
for reach in [0.40,0.45,0.50,0.56,0.60,0.65]:
    qa,_=ik(np.array(HOME,float),np.array([reach,0,0.623-0.3947]),DOWN_R,tool=0.0,iters=300)
    qb,_=ik(qa,np.array([reach,0,0.75-0.3947]),DOWN_R,tool=0.0,iters=300)
    print(reach, np.round(np.abs(qb-qa).max(),3), np.round(qb-qa,2))
