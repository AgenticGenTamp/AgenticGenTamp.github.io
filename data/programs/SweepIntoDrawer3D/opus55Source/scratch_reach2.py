import numpy as np
from approach import Kin, _get_model, R_down, Q_HOME
k=Kin(_get_model())
for bx in [1.38,1.42]:
  for x in [0.58,0.62,0.66,0.7,0.8,0.94]:
    for t in [0.0,0.2,0.4]:
      q=k.ik_multi(np.array([bx,0,np.pi]),Q_HOME,np.array([x,0,0.475]),R_down(t))
      q2=k.ik_multi(np.array([bx,0,np.pi]),Q_HOME,np.array([x,0.15,0.475]),R_down(t))
      print(bx,x,t,q is not None, q2 is not None)
