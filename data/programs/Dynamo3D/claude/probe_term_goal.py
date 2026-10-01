import sys, numpy as np
from env_client import make_env
from probe_term_lib import *
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
cn=chairs(obs)[0]; c0=cxy(obs,cn)
# 1) push chair +x away from goal area, stop early
obs,t,_=move_to(env,obs,c0+np.array([-1.0,0.0]))
for i in range(200):
    obs,rew,t,tr,info=env.step(act(0.03,0.))
    if t or cxy(obs,cn)[0]-c0[0]>0.30: break
print('phase1 term',t,'chair',np.round(cxy(obs,cn),3),'robot',np.round(rxy(obs),3),flush=True)
# 2) retreat and go far south
obs,t,_=move_to(env,obs,np.array([0.2,-1.2])); print('retreat term',t,flush=True)
# 3) approach a vertical line x=0.95 from y=-1.2 upward
obs,t,_=move_to(env,obs,np.array([0.95,-1.2])); print('to south point term',t,'robot',np.round(rxy(obs),3),flush=True)
for i in range(60):
    rp=rxy(obs)
    obs,rew,t,tr,info=env.step(act(0.,0.02))
    if t:
        print(f'TERM crossing north: rprev={np.round(rp,3)} rend={np.round(rxy(obs),3)} chair={np.round(cxy(obs,cn),3)}'); break
else: print('no term up to',np.round(rxy(obs),3),'chair',np.round(cxy(obs,cn),3))
env.close()
