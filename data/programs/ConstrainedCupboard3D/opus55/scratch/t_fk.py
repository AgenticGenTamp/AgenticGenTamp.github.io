import sys; sys.path.insert(0,'.')
from kin import fk
import numpy as np
q=[0,-0.349,3.142,-2.548,0,-0.873,1.571]
T=fk(q,0.0); print(np.round(T,3))
q=[0,0,0,0,0,0,0]; print(np.round(fk(q),3))
