import time, numpy as np
from env_client import make_env
from approach import GeneratedApproach
env=make_env(); o,info=env.reset(seed=1)
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(o,info)
ts=[]
for i in range(300):
    t0=time.time(); a=ap.get_action(o); ts.append(time.time()-t0)
    o,r,t,tr,info=env.step(a)
    if t: break
ts=np.array(ts)
print("n",len(ts),"total",round(ts.sum(),2),"mean",round(ts.mean()*1000,1),"max",round(ts.max()*1000,1),"p95",round(np.percentile(ts,95)*1000,1))
env.close()
