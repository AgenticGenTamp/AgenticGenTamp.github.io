import numpy as np
from approach import *
from approach import _get_model
K=Kin(_get_model())
def Rfront(psi):  # approach toward -x, pitched down by psi; fingers close in x-z plane
    z=np.array([-np.cos(psi),0,-np.sin(psi)]); y=np.array([0,1.,0]); x=np.cross(y,z)
    return np.column_stack([x,y,z])
rng=np.random.default_rng(0)
seeds=[Q_HOME,Q_RETRACT]+[rng.uniform(-np.pi,np.pi,7)*np.array([1,.7,1,.8,1,.65,1]) for _ in range(30)]
for psi in [0,0.3,0.5,0.7,0.9]:
  for by in [0,-0.2,-0.35]:
    s=''
    for bx in [1.22,1.26,1.3,1.35,1.4]:
        found=0
        for sd in seeds:
            q,ok=K.ik(np.array([bx,by,np.pi]),sd,np.array([0.934,0,0.325]),Rfront(psi),iters=100)
            if ok: found=1;break
        s+=str(found)
    print(psi,by,s)
