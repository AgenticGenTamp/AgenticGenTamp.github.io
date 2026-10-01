import time, numpy as np
from ik10 import solve, tcp
from ik import down_R
c0=np.array([-0.12,0,0, 0,-0.35,-np.pi,-2.5,0,-0.87,np.pi/2])
for use_base in [False, True]:
  for tgt in [[0.3,0.3,0.15],[0.2,-0.33,0.15],[0.3,0.0,0.2]]:
    t=time.time(); c,md,err=solve(c0,np.array(tgt),down_R(0),yaw_free=True,use_base=use_base)
    print(use_base,tgt,'maxd',round(md,3),'steps',int(np.ceil(md/0.2-1e-6)),'err',err,'t',round(time.time()-t,3), np.round(c[:3],3))
