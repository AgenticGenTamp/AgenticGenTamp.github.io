import time, numpy as np
from env_client import make_env
from approach import GeneratedApproach
env=make_env(); obs,info=env.reset(seed=0)
ap=GeneratedApproach(env.action_space, env.observation_space, {})
t0=time.time(); ap.reset(obs,info); tr=time.time()-t0
tot=0.0; n=0
for t in range(700):
    t0=time.time(); a=ap.get_action(obs); tot+=time.time()-t0; n+=1
    obs,r,te,tru,_=env.step(a)
    if te: break
print(f"reset {tr*1000:.1f}ms  policy total {tot:.3f}s over {n} steps ({tot/n*1000:.2f} ms/step)")
env.close()
