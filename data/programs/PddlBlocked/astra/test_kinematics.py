import time
import numpy as np
from kinematics import fk,solve

rng=np.random.default_rng(4)
start=time.perf_counter()
errors=[]
for i in range(30):
 yaw=rng.uniform(-np.pi,np.pi); base=rng.uniform(-3,3,2)
 q=np.array([base[0],base[1],yaw,0.,rng.uniform(.44,.65),0.,rng.uniform(-.5,-.3),0.,0.,0.])
 q[8]=-q[4]-q[6]
 if q[8]>0:q[7]=np.pi;q[8]=-q[8];q[9]=np.pi
 target,R=fk(q)
 result,error=solve(target,yaw,base)
 p,R2=fk(result)
 errors.append(error)
 assert error<1e-3,(i,error,target,result)
print('Thirty reachable targets: max error',max(errors),'mean seconds',(time.perf_counter()-start)/30)
for target,base,yaw in [([4.4,.1,.82],[3.6,0.],0.),([4.4,.1,.82],[4.0,-.7],.8),([4.4,.1,.82],[4.8,-.7],2.1)]:
 t=time.perf_counter();q,error=solve(target,yaw,base);print('Test',base,yaw,'error',error,'seconds',time.perf_counter()-t,'q',np.round(q,3).tolist())
