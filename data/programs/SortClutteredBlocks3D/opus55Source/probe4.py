import numpy as np
from kin import Kin, down_rot
K=Kin()
base=np.array([0.6,0,np.pi])
seeds=[np.array([0,0.26,3.14,-2.27,0,0.96,1.57]), np.array([0,-0.35,3.14,-2.55,0,-0.87,1.57]), np.zeros(7), np.array([0,0.5,0,1.5,0,1.0,0])]
for sd in seeds:
    for tgt in [[0,0,0.55],[0,0,0.43],[-0.1,0.15,0.5],[-0.13,-0.18,0.5],[0.1,0.15,0.5]]:
        q,e=K.ik(base,sd,np.array(tgt),down_rot(0.0),iters=300)
        print(np.round(sd[:2],2),tgt,round(e,4),q.round(2))
