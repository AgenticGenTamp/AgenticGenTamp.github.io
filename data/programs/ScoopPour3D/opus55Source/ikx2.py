import numpy as np, sys
from approach import Kin, down_R, axis_R
k=Kin()
c=np.array([0.5,0.2,0.72]); Gp=np.array([-0.215,0,0.021])
R0=down_R(0.0)
by=float(sys.argv[1])
q=np.array([-1.47,1.94,4.33,-1.69,0.89,-1.77,-0.42]); base=np.array([-0.16,by,0])
for phi in np.radians(np.arange(-165,5,15)):
    Rb=axis_R(np.array([1.,0,0]),phi)
    q,e=k.ik(base,q,c+Rb@Gp,Rb@R0,iters=200)
    print(round(np.degrees(phi)),round(e,4),np.round(q,2))
# also grasp pose at source
q=np.array([0,-0.349,3.142,-2.548,0,-0.873,1.571])
for z in [0.56,0.5]:
  q,e=k.ik(np.array([-0.16,-0.2,0]),q,np.array([0.285,-0.2,z]),R0,iters=200); print('src',z,round(e,4),np.round(q,2))
