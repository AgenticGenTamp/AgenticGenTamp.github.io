import numpy as np
from colcheck import k, contacts, path
from approach import tilt_rot, RETRACT
bt=np.array([0,0,0.])
for R in [0.45,0.55,0.65,0.75]:
  for al in [0.7,0.9]:
    c=np.array([R+0.12,0,0.025]); Rg=tilt_rot(0,al)
    tgt=c-0.12*Rg[:,2]+[0,0,0.03]
    res=set()
    for j2 in [0.0,0.4,0.8,1.2]:
      for j4 in [-2.2,-1.6,-1.0]:
        for j6 in [-1.5,-0.8,0.0,0.8]:
            q0=np.array([0,j2,np.pi,j4,0,j6,np.pi/2])
            qa,err=k.ik(bt,q0,tgt,Rg,qnom=q0,null_w=0.0)
            if err<1e-3:
                qb,err2=k.ik(bt,qa,c+0.01*Rg[:,2],Rg,qnom=qa,null_w=0.0)
                colb=[x for x in contacts(bt,qb) if 'pad' not in x[1] and 'follower' not in x[1]]
                res.add((len(contacts(bt,qa))==0, len(path(bt,RETRACT,qa))==0, len(colb)==0, tuple(qa.round(1))))
    print(R,al); [print('   ',r) for r in sorted(res)]
