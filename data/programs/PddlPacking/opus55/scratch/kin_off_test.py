import sys; sys.path.insert(0,'.')
from kin import *
q0=np.array([0.6772,-0.3431,1.2,-1.4669,1.2422,-1.9544,2.2225])
# check wrist roll rotates about tool x axis for top-down
t1,R1=fk((-0.5,0,0),q0); q=q0.copy(); q[6]+=np.pi/2; t2,R2=fk((-0.5,0,0),q)
print(t1,t2,R1[:,1],R2[:,1])
