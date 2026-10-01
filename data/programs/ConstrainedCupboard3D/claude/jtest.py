import numpy as np, kin, ctrl
from approach import GeneratedApproach
from env_client import make_env
class FA: low=-np.ones(11,dtype=np.float32)*0.1; high=np.ones(11,dtype=np.float32); 
env=make_env(); o,i=env.reset(seed=0,options={'object_count':1})
ap=GeneratedApproach(FA(),None,{}); ap.reset(o,i)
qd=ap._cfg('hover'); print("qgoal",np.round(qd,3))
for t in range(300):
    q,b=ctrl.read_robot(o)
    a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip((qd-q)/0.25,-0.1,0.1)
    o,r,te,tr,inf=env.step(a)
    if t%60==0: print(t,np.round(qd-q,3))
q,b=ctrl.read_robot(o)
print("final err",np.round(qd-q,4))
print("ee",np.round(ctrl.ee_world(q,b)[:3,3]-np.array([b[0],b[1],0]),4))
env.close()
