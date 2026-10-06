import numpy as np
from kin import *
env,m=make_model(); K=Kin(m)
Rh=np.column_stack([[0,0,1],[0,1,0],[-1,0,0]])
for bx in [1.25,1.3,1.35,1.4]:
    q=K.ik_multi(np.array([bx,0,np.pi]),Q_HOME,np.array([0.935,0,0.325]),Rh)
    q2=K.ik_multi(np.array([bx,0,np.pi]),Q_HOME,np.array([1.03,0,0.325]),Rh)
    print(bx, None if q is None else q.round(2), None if q2 is None else q2.round(2))
