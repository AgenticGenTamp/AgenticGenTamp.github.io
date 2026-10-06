import numpy as np
from tosslib import Kin
k=Kin()
for j7 in [0, np.pi/2]:
    q=[0,-1.0,np.pi,0.6,0,0.6,j7]
    p,R=k.fk([0,0,0],q); print(j7, p.round(3), 'x', R[:,0].round(2), 'z', R[:,2].round(2))
