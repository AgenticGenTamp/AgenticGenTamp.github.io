import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); t0=int(sys.argv[2]); t1=int(sys.argv[3]); every=int(sys.argv[4]) if len(sys.argv)>4 else 3
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(t1):
    a=ap.get_action(obs); obs,rew,term,trunc,_=env.step(a)
    ro=obs.get_object_from_name('robot'); ho=obs.get_object_from_name('hook')
    if t>=t0 and t%every==0: print(t, ap.phase, np.round(a,3), [round(obs.get(ro,f),3) for f in ('x','y','theta','arm_joint','finger_gap')], 'hook',[round(obs.get(ho,f),2) for f in ('x','y','theta','held')])
    if term: break
