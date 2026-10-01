import sys; sys.path.insert(0,'.')
from kin import *
q0=np.array([0,-0.35,np.pi,-2.55,0,-0.87,np.pi/2])
for name,R in [('v+',np.column_stack([[0,1,0],[0,0,1],[1,0,0]])),('v-',np.column_stack([[0,-1,0],[0,0,-1],[1,0,0]]))]:
  for reach in [0.45,0.55,0.65]:
    for z in [-0.15,-0.1,-0.05, 0.0]:
        q,ok=ik_best(q0,np.array([reach,0,z]),R,tool=0.15,n_random=40)
        print(name,reach,z,ok,np.round(q,2))
