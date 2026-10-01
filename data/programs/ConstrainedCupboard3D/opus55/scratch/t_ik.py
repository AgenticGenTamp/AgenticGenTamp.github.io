import sys, time; sys.path.insert(0,'.')
from kin import *
q0=np.array([0,-0.349,3.142,-2.548,0,-0.873,1.571])
Rd=np.array([[1,0,0],[0,-1,0],[0,0,-1.]])
t=time.time()
for p in [[0.5,0,-0.3],[0.6,0.2,-0.25],[0.4,-0.2,0.0]]:
    q,e1,e2=ik(q0,np.array(p),Rd,tool=0.14,q_nom=q0)
    print(np.round(q,2),e1,e2)
print(time.time()-t)
