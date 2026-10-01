import sys; sys.path.insert(0,'.')
from kin import *
import time
q0=np.array([0.6772,-0.3431,1.2,-1.4669,1.2422,-1.9544,2.2225])
print(fk((-1,0,0),q0))
t=time.time(); oks=0
for i in range(50):
    q,ok=ik_down((-0.5,0,0),np.array([0.2*np.cos(i),0.3*np.sin(i),0.79]),q0,yaw=i*0.3)
    oks+=ok
print((time.time()-t)/50, oks)
