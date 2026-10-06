import numpy as np
from kin import *
env,m=make_model(); K=Kin(m)
def Rpitch(th):  # pointing down, tilted so approach has -x component; fingers along y
    z=np.array([-np.sin(th),0,-np.cos(th)]); x=np.array([0,1.,0]); y=np.cross(z,x)
    return np.column_stack([x,y,z])
for bx in [1.33,1.4]:
  for th in [0,0.3,0.6]:
    row=[]
    for x in [0.55,0.6,0.7,0.8,0.9,1.0]:
        s=''
        for y in [-0.45,-0.3,-0.1,0.05]:
            q=K.ik_multi(np.array([bx,0,np.pi]),Q_HOME,np.array([x,y,0.5]),Rpitch(th))
            s+='1' if q is not None else '0'
        row.append(s)
    print(bx,th,row)
