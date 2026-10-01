import numpy as np, time
from env_client import make_env
import approach as A
env=make_env(); obs,info=env.reset(seed=1, options={'object_count':20})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); 
t0=time.time(); ap.reset(obs,info)
pol=0.0
for t in range(1000):
    s=time.time(); a=ap.get_action(obs); pol+=time.time()-s
    obs,r,te,tr,i=env.step(a)
    if te: print('TERM',t); break
print('policy cpu time', round(pol,2), 'wall', round(time.time()-t0,1), 'cubes done idx', ap.idx)
cb={n:np.round(A.obj_pos(obs,n),3) for n in obs.get_object_names() if n.startswith('cube')}
inbin=0
for n,p in cb.items():
    for bn,bp in ap.bins.items():
        if np.linalg.norm(p[:2]-bp[:2])<0.06: inbin+=1
print('cubes in some bin:',inbin,'of',len(cb))
env.close()
