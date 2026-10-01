import sys; sys.path.insert(0,'.')
from pr2fk import *
import time
q0=np.array([0.6772,-0.3431,1.2,-1.4669,1.2422,-1.9544,2.2225])
xa=np.array([0,0,-1.]); ya=np.array([1,0,0.]); Rd=np.stack([xa,ya,np.cross(xa,ya)],1)
t=time.time()
for i in range(20):
    q,e=ik((-0.6,0,0),np.array([0.1*np.cos(i),0.2*np.sin(i),0.79]),Rd,q0)
print((time.time()-t)/20, e)
