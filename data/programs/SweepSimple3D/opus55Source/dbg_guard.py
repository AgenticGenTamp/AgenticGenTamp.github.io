import numpy as np
from approach import GeneratedApproach
b=np.array([1.098,1.55,-2.074])
H=GeneratedApproach._arm_hits
print('cur',H(b[:2],b[2]))
for dr in [0.1,0.05,0.02]:
    print(dr,H(b[:2],b[2]+dr))
print('north',H(b[:2]+[0,0.1],b[2]), H(b[:2]+[0,0.1],b[2]+0.1))
