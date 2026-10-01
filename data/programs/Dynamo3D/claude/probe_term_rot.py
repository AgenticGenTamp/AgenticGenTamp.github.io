import sys, numpy as np
from env_client import make_env
from probe_term_lib import *
seed=int(sys.argv[1]); nrot=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed)
for i in range(nrot):
    obs,rew,t,tr,info=env.step(act(0,0,0.1))
r=obs.get_object_from_name('robot'); print('rot',round(float(obs.get(r,'pos_base_rot')),3),flush=True)
obs,t,_=move_to(env,obs,np.array([1.0,-1.1]),maxsteps=150)
print('to start term',t,flush=True)
for i in range(80):
    rp=rxy(obs)
    obs,rew,t,tr,info=env.step(act(0,0.02))
    if t: print(f'rot={round(float(obs.get(obs.get_object_from_name("robot"),"pos_base_rot")),2)} CROSS prev={np.round(rp,3)} end={np.round(rxy(obs),3)}'); break
else: print('noterm',np.round(rxy(obs),3))
env.close()
