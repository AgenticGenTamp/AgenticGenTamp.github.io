import numpy as np
from kin import *
env,m=make_model(); K=Kin(m)
def Rtilt(th, fingers_y=True):  # approach down, top leaning +x by th
    z=np.array([-np.sin(th),0,-np.cos(th)]); x=np.array([0,1.,0]); y=np.cross(z,x)
    return np.column_stack([x,y,z])
for bx in [1.25,1.3,1.35,1.4]:
  for th in [0.15,0.25]:
    q=K.ik_multi(np.array([bx,0,np.pi]),Q_HOME,np.array([0.912,0,0.335]),Rtilt(th))
    print(bx,th, None if q is None else q.round(2))
