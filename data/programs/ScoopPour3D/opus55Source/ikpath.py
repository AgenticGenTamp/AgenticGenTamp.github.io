import numpy as np, sys
from approach import Kin, down_R, axis_R, ARM_LO, ARM_HI
K=Kin()
base=np.array([-0.16, 0.2+0.14-0.1, 0.0])
q0=np.array([0, -0.349, 3.142, -2.548, 0, -0.873, 1.571])
R0=down_R(np.pi/2); Gp=np.array([0,-0.14,0.021]); Gr=R0
c1=np.array([0.5,0.2,0.72])
# grasp config
q,_=K.ik(np.array([-0.16,-0.44,0]), q0, np.array([0.5,-0.34,0.515]), R0, iters=200)
print('grasp q', np.round(q,2))
for phi in np.linspace(0,-np.radians(165),12):
    Rb=axis_R(np.array([1.,0,0]),phi)
    pos=c1+Rb@Gp; R=Rb@Gr
    base=np.array([-0.16, pos[1]-0.1, 0])
    q,err=K.ik(base,q,pos,R,iters=200)
    print(round(np.degrees(phi)), 'err',round(err,4),'q',np.round(q,2))
