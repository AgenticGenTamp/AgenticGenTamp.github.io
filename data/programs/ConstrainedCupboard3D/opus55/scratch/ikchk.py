import sys; sys.path.insert(0,'.')
from kin import *
q0=np.array([0,-0.35,np.pi,-2.55,0,-0.87,np.pi/2])
for name,R in [('v+',np.column_stack([[0,1,0],[0,0,1],[1,0,0]])),('v-',np.column_stack([[0,-1,0],[0,0,-1],[1,0,0]])),('down',np.array([[1,0,0],[0,-1,0],[0,0,-1.]]))]:
  for reach in [0.4,0.5,0.6]:
    for z in [-0.3,-0.1,0.1]:
        best=None
        for s in [q0]+list(np.random.default_rng(0).uniform(-2,2,(20,7))):
            q,ep,er=ik(s,np.array([reach,0,z]),R,tool=0.15,iters=200)
            if ep<2e-3 and er<2e-2:
                m=max(abs(q[1])/2.0,abs(q[3])/2.45,abs(q[5])/1.95)
                if best is None or m<best[0]: best=(m,np.round(q,2))
        print(name,reach,z,best)
