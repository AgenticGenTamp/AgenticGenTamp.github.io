import numpy as np
from approach import Kin, down_R, axis_R
k=Kin()
R0=down_R(np.pi/2)  # dir=+y
ax=np.array([1.,0,0])
res=[]
for zc in [0.62,0.67,0.72]:
 for gx in [-0.08,-0.04,0.0,0.04,0.08]:
  for boff in [-0.25,-0.15,-0.05,0.05]:
    c=np.array([0.5,0.2,zc]); Gp=np.array([gx,-0.14,0.021])
    q=np.array([0,-0.349,3.142,-2.548,0,-0.873,1.571]); worst=0; mxe=0; j4=9
    for phi in np.radians(np.arange(0,-170,-10)):
        Rb=axis_R(ax,phi); pos=c+Rb@Gp
        base=np.array([-0.16,pos[1]+boff,0])
        for _ in range(3): q,e=k.ik(base,q,pos,Rb@R0,iters=100)
        mxe=max(mxe,e); worst=max(worst,abs(q[5])/2.09,abs(q[1])/2.24,abs(q[3])/2.57); j4=min(j4,abs(q[3]))
    res.append((round(mxe,3),round(worst,2),round(j4,2),zc,gx,boff))
for r in sorted(res,key=lambda r:(r[0]>0.005,r[1]))[:15]: print(r)
