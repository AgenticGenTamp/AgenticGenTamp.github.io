import numpy as np, sys, kin
from env_client import make_env
from approach import GeneratedApproach
seed,cnt,T=3,10,232
env = make_env(); obs, info = env.reset(seed=seed, options={'object_count':cnt})
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for t in range(T):
    a = ap.get_action(obs); obs, r, term, trunc, info = env.step(a)
ap._read(obs)
print('base',ap.base, 'q',ap.q.round(3)); p,Rm=kin.fk(*ap.base,ap.q); print('tip3d',p.round(3))
for n,c in ap.cubes.items(): print(n,c[:3].round(3))
print('others',{k:v.round(2) for k,v in ap.others.items()})
R=obs.get_object_from_name('robot'); print([round(obs.get(R,f),3) for f in ['vel_base_x','vel_base_y','pos_gripper']])
