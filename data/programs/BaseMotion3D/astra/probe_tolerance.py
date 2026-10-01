import numpy as np
from env_client import make_env
cases=[(.01,0),(.04,0),(.05,0),(.06,0),(.1,0),(.2,0),(.3,0),(.4,0),(.5,0),(.1,.1),(.2,.2),(.3,.3)]
for offset in cases:
    env=make_env(); s,_=env.reset(seed=0)
    goal=s[19:21]+np.array(offset,dtype=np.float32)
    term=False
    for step in range(8):
        a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(goal-s[:2],-.4,.4)
        s,r,term,tr,info=env.step(a)
        if term or tr or np.linalg.norm(goal-s[:2])<1e-6:break
    print('OFFSET',offset,'step',step+1,'residual',(s[:2]-s[19:21]).tolist(),'term',term,flush=True)
    env.close()
for extra in ('rotate','joints','gripper'):
    env=make_env();s,_=env.reset(seed=0)
    for step in range(8):
        a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(s[19:21]-s[:2],-.4,.4)
        if extra=='rotate':a[2]=.4
        if extra=='joints':a[3:10]=.4
        if extra=='gripper':a[10]=-1
        s,r,term,tr,info=env.step(a)
        if term or tr:break
    print('EXTRA',extra,'step',step+1,'residual',(s[:2]-s[19:21]).tolist(),'term',term,flush=True)
    env.close()
for offset in ((.035,.035),(.04,.04),(.049,.049),(.051,0),(.05,.001)):
    env=make_env();s,_=env.reset(seed=0);goal=s[19:21]+offset
    for step in range(8):
        a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(goal-s[:2],-.4,.4)
        s,r,term,tr,info=env.step(a)
        if term or tr or np.linalg.norm(goal-s[:2])<1e-6:break
    print('BOUNDARY',offset,'residual',(s[:2]-s[19:21]).tolist(),'term',term,flush=True)
    env.close()
