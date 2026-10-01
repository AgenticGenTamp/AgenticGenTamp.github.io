import numpy as np, time
from env_client import make_env
import approach as A
env=make_env(); ap=A.GeneratedApproach(env.action_space, env.observation_space, {})
tot=0; n=0; mx=0
for seed in range(3):
    obs,info=env.reset(seed=seed, options={'object_count':4})
    ap.reset(obs,info)
    for t in range(1000):
        t0=time.time(); a=ap.get_action(obs); dt=time.time()-t0
        tot+=dt; n+=1; mx=max(mx,dt)
        obs,r,term,tr,info=env.step(np.asarray(a,dtype=np.float32))
        if term: break
print("mean per-step %.4f s, max %.4f s, total per-episode %.2f s"%(tot/n, mx, tot/3))
env.close()
