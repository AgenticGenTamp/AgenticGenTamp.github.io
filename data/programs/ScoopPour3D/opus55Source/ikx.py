import numpy as np, sys
from approach import Kin, down_R, axis_R
k=Kin()
c=np.array([0.5,0.2,0.72]); Gp=np.array([-0.215,0,0.021])
R0=down_R(0.0)
for by in [0.2,0.1,0.0]:
  q=np.array([0,-0.349,3.142,-2.548,0,-0.873,1.571]); base=np.array([-0.16,by,0])
  print("base y",by)
  for phi in np.radians(np.arange(0,-170,-15)):
    Rb=axis_R(np.array([1.,0,0]),phi)
    q,e=k.ik(base,q,c+Rb@Gp,Rb@R0,iters=200)
    print(round(np.degrees(phi)),round(e,4),np.round(q,2))
