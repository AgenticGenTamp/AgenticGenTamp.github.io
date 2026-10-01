import numpy as np, kin, ctrl
from approach import GeneratedApproach
from env_client import make_env
class FA:
    low=-np.ones(11,dtype=np.float32)*0.1; high=np.ones(11,dtype=np.float32)*0.1
env=make_env(); o,i=env.reset(seed=0,options={'object_count':1})
ap=GeneratedApproach(FA(),None,{}); ap.reset(o,i)
ap._set_goal('hover',0.13)
for t in range(200):
    q,b=ctrl.read_robot(o)
    dq,done=ap._arm_cmd(q)
    a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(dq/0.25,-0.1,0.1)
    if t>190: print(t,"q",np.round(q,3),"cmd",np.round(a[3:10],3))
    o,r,te,tr,inf=env.step(a)
q,b=ctrl.read_robot(o); print("ee",np.round(kin.arm_fk(q,ctrl.TOOL)[:3,3],4),"goal",np.round(ap.T_goal[:3,3],4))
env.close()
