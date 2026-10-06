from ik_minmax import ikmm
from ik_null import ikn, Rd
from ik_tool import *
import numpy as np
R0=0.5
q5,ep,er=ikmm(np.array([R0,0,0.05]),Rd)
zs=[round(0.02+0.01*i,2) for i in range(9)]
tab={0.05:q5}
for direction in [ [0.04,0.03,0.02], [0.06,0.07,0.08,0.09,0.10] ]:
    prev=q5
    for z in direction:
        q,ep,er=ikn(np.array([R0,0,z]),Rd,prev,prev); assert ep<1e-3 and er<1e-3,(z,ep,er)
        tab[z]=q; prev=q
print('HOLD_R =',R0)
print('ARM_TABLE_Z =',zs)
print('ARM_TABLE_Q = [')
for z in zs: print('    %s,'%np.round(tab[z],5).tolist())
print(']')
