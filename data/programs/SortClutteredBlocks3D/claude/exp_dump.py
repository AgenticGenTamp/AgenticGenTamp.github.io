import numpy as np, sys
from env_client import make_env
import approach as A
env=make_env(); obs,info=env.reset(seed=0, options={'object_count':4})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for T in range(1,401):
    obs,r,te,tr,i=env.step(ap.get_action(obs))
    if te or tr: print('TERM',T); break
o=obs.get_object_from_name('cube1')
print('cube feats', {f: round(float(obs.get(o,f)),4) for f in ['bb_x','bb_y','bb_z']})
ob=obs.get_object_from_name('bin_red')
print('bin feats', {f: round(float(obs.get(ob,f)),4) for f in ['bb_x','bb_y','bb_z']})
for n in ['cube1','cube2','cube3','cube4']:
    c=A.obj_pos(obs,n); bp=A.obj_pos(obs,A.BIN_ORDER[(int(n[4:])-1)%4])
    print(n, np.round(c,4).tolist(), 'bin', np.round(bp,4).tolist(), 'd', np.round(c-bp,4).tolist())
