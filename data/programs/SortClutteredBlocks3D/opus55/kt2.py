import numpy as np
from kin import ik, fk_arm
np.set_printoptions(precision=3,suppress=True)
D=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
home=[0,-0.349,3.142,-2.548,0,-0.873,1.571]
for r in [0.42,0.48,0.52]:
    qc,e=ik(home,np.array([r,0,0.36]),D,tool=0,iters=300)
    qg,e2=ik(qc,np.array([r,0,0.229]),D,tool=0,iters=300)
    print(r,qc,e,qg,e2,"maxdiff",np.max(np.abs(qg-qc)), "from home", np.max(np.abs(qc-np.array(home))))
