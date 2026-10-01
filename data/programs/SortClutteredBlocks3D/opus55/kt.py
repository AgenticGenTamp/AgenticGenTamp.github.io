import numpy as np
from kin import fk_arm
q=[0,-0.349,3.142,-2.548,0,-0.873,1.571]
tip,R,ps,ax=fk_arm(q,0.2)
np.set_printoptions(precision=3,suppress=True)
print(tip); print(R)
q=[0,0,0,0,0,0,0]
print(fk_arm(q,0.)[0])
