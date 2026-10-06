import numpy as np
from colcheck import k, contacts, path
from approach import tilt_rot, RETRACT
bt=np.array([0,0,0.]); c=np.array([0.57,0,0.025]); Rg=tilt_rot(0,0.9)
tgt=c-0.12*Rg[:,2]+[0,0,0.03]
rng=np.random.default_rng(0)
sols=[]
for j2 in [0.0,0.4,0.8,1.2]:
  for j4 in [-2.2,-1.6,-1.0]:
    for j6 in [-1.5,-0.8,0.0,0.8]:
        q0=np.array([0,j2,np.pi,j4,0,j6,np.pi/2])
        qa,err=k.ik(bt,q0,tgt,Rg,qnom=q0,null_w=0.0)
        if err<1e-3:
            col=contacts(bt,qa); pc=path(bt,RETRACT,qa)
            sols.append((len(col)==0, len(pc)==0, qa.round(2).tolist()))
for s in sorted(set(map(str,sols))): print(s)
