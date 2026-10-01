import numpy as np
from kin import *
q0=np.array([0,-0.349,3.142,-2.548,0,-0.873,1.571])
pts=[(x,y,z) for x in [0.62,0.72,0.82] for y in [-0.2,-0.1,0.0] for z in [0.455,0.53]]+[(x,y,0.47) for x in [1.0,1.1] for y in [-0.15,0,0.15]]
for bx in [1.15,1.25]:
  for by in [-0.6,-0.7]:
    for th in [np.pi, np.pi-0.5, np.pi-0.8]:
      errs=[]
      for p in pts:
        best=9
        for yaw in [0,np.pi/4,np.pi/2,-np.pi/4]:
          q,e=ik((bx,by,th),q0,np.array(p),R_down(th+yaw),iters=200); best=min(best,e)
        errs.append(best)
      print(bx,by,round(th,2), "max err %.4f"%max(errs), "nfail",sum(e>1e-3 for e in errs))
