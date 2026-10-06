import numpy as np
from tosslib import Kin
k=Kin()
q=[0,-0.349,3.142,-2.548,0,-0.873,1.571]
p,R=k.fk([0,0,0],q); print(p); print(R.round(3))
q=[0,0,0,0,0,0,0]
p,R=k.fk([0,0,0],q); print(p); print(R.round(3))
