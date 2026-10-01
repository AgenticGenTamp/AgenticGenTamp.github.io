import numpy as np, sys
from env_client import make_env
from approach import *
env=make_env(); seed=int(sys.argv[1]); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
name=ap.fast[0]; print('target',name, 'plan', [np.round(a,2) for a in ap.fast[1]])
c=obs.get_object_from_name(name); r=obs.get_object_from_name('robot')
for t in range(20):
    a=ap.get_action(obs); obs,rw,te,tr,_=env.step(a)
    print(t,'fast' if ap.fast is not None else 'fb','cz',round(obs.get(c,'pose_z'),3),'ga',obs.get(r,'grasp_active'),te)
    if te: break
env.close()
