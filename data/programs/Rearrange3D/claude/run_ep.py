import numpy as np,sys,time
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); MAX=int(sys.argv[2]) if len(sys.argv)>2 else 700
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space,env.observation_space,{})
ap.reset(obs,info)
o=np.asarray(obs); bowl=o[0:3]
print("seed",seed,"bowl",np.round(bowl,3),"drink",np.round(o[16:19],3),"can",np.round(o[32:35],3))
t0=time.time(); tot=0
for t in range(MAX):
    a=ap.get_action(obs)
    obs,r,te,tr,inf=env.step(a); tot+=r
    if te or tr: break
o=np.asarray(obs)
print("steps",t+1,"term",te,"tot%.1f"%tot,"wall%.1f"%(time.time()-t0))
print(" final bowl",np.round(o[0:3],3),"drink",np.round(o[16:19],3),"can",np.round(o[32:35],3))
print(" d_drink %.3f d_can %.3f"%(np.linalg.norm(o[16:18]-o[0:2]),np.linalg.norm(o[32:34]-o[0:2])))
env.close()
