from fk import fk
import numpy as np
q=[0,-0.35,-np.pi,-2.5,0,-0.87,np.pi/2]
M,fr=fk(q,base=(-0.12,0,0))
for f in fr: print(np.round(f[:3,3],3))
print(np.round(M[:3,:3],3))
