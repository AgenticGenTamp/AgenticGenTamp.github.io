import time, numpy as np
from ik import ik_pose, down_R
from fk import fk
q=np.array([0,-0.35,-np.pi,-2.5,0,-0.87,np.pi/2]); base=np.array([-0.12,0,0])
Mt=np.eye(4); Mt[:3,:3]=down_R(0); Mt[:3,3]=[0.3,0.3,0.2]
t=time.time(); 
for _ in range(5): r=ik_pose(Mt,q,base=base)
print('far ik', (time.time()-t)/5, r[1])
q2=r[0]; Mt[2,3]=0.18
t=time.time();
for _ in range(20): r=ik_pose(Mt,q2,base=base)
print('near ik', (time.time()-t)/20, r[1])
t=time.time();
for _ in range(1000): fk(q,base)
print('fk', (time.time()-t)/1000)
