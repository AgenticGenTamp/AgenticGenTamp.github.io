import sys; sys.path.insert(0,'.')
import numpy as np
from kin import ik_best, fk
MZ=0.3949
home=np.array([0,-0.349,3.142,-2.548,0,-0.873,1.571])
for a in [20,30,45,60,75,90]:
    ar=np.radians(a); z=np.array([np.cos(ar),0,-np.sin(ar)]); x=np.array([0,1.,0]); y=np.cross(z,x)
    R=np.column_stack([x,y,z])
    for reach in [0.3,0.4,0.5,0.6,0.7]:
        for tool in [0.15,0.20]:
            q,ok=ik_best(home,np.array([reach,0,0.03-MZ]),R,tool=tool,n_random=20)
            if ok: print(a,reach,tool,np.round(q,2))
