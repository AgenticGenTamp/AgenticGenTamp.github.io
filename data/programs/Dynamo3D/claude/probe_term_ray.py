import sys, numpy as np
from env_client import make_env
from probe_term_lib import *
# args: seed  x0,y0  dx,dy  [approach_via]
seed=int(sys.argv[1]); start=np.array([float(v) for v in sys.argv[2].split(',')])
d=np.array([float(v) for v in sys.argv[3].split(',')]); d=d/np.linalg.norm(d)
env=make_env(); obs,info=env.reset(seed=seed)
cns=chairs(obs)
obs,t,_=move_to(env,obs,start,maxsteps=120)
if t: print(f'ray {start} {d}: TERM while going to start at {np.round(rxy(obs),3)}'); env.close(); sys.exit()
for i in range(150):
    rp=rxy(obs)
    obs,rew,t,tr,info=env.step(act(d[0]*0.02,d[1]*0.02))
    if t:
        print(f'ray start={start} dir={np.round(d,2)} CROSS rprev={np.round(rp,3)} rend={np.round(rxy(obs),3)}'); break
else:
    print(f'ray start={start} dir={np.round(d,2)} NOTERM end={np.round(rxy(obs),3)}')
env.close()
