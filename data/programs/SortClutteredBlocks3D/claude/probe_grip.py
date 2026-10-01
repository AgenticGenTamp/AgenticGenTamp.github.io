import numpy as np
from probe_act import rollout, FEATS
for g in (0.0,0.5,1.0):
    a=[0.0]*10+[g]
    h,_,_,_,_=rollout(a,steps=40)
    print(f"grip={g}: pos_gripper traj",np.round(h[[0,1,2,3,5,10,20,39],10],4))
