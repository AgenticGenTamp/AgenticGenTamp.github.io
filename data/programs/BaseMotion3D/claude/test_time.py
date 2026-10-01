import numpy as np, time
from env_client import make_env
from approach import GeneratedApproach
env=make_env()
for s in [72,560,601,779,0,5]:
    o,i=env.reset(seed=s)
    ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(o,i)
    t0=time.time(); pol=0.0
    for k in range(env.max_steps):
        a=time.time(); act=ap.get_action(o); pol+=time.time()-a
        o,r,t,tr,_=env.step(act)
        if t or tr: break
    print(s,"steps",k+1,"policy time",round(pol,3),"total",round(time.time()-t0,2),"term",t)
