import sys, numpy as np
from env_client import make_env
from probe_term_lib import *
seed=int(sys.argv[1]); cnt=int(sys.argv[2]); start=np.array([float(v) for v in sys.argv[3].split(',')])
d=np.array([float(v) for v in sys.argv[4].split(',')]); d=d/np.linalg.norm(d)
env=make_env(); obs,info=env.reset(seed=seed,options={'object_count':cnt})
obs,t,_=move_to(env,obs,start,maxsteps=200)
if t: print(f'c{cnt} s{seed} ray {start} {d}: TERM en route at {np.round(rxy(obs),3)}'); env.close(); sys.exit()
for i in range(200):
    rp=rxy(obs)
    obs,rew,t,tr,info=env.step(act(d[0]*0.02,d[1]*0.02))
    if t:
        print(f'c{cnt} s{seed} start={start} dir={np.round(d,2)} CROSS prev={np.round(rp,3)} end={np.round(rxy(obs),3)}'); break
else: print(f'c{cnt} s{seed} start={start} dir={np.round(d,2)} NOTERM end={np.round(rxy(obs),3)}')
env.close()
