import sys, numpy as np
from env_client import make_env
sys.path.insert(0,'.')
from approach import GeneratedApproach
s=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=s)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
def rb(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,'pos_base_x')),float(o.get(r,'pos_base_y'))])
def ch(o):
    return {n:np.round([float(o.get(o.get_object_from_name(n),'x')),float(o.get(o.get_object_from_name(n),'y'))],2).tolist() for n in sorted(o.get_object_names()) if n!='robot'}
print("start",np.round(rb(obs),2),ch(obs))
for i in range(400):
    a=ap.get_action(obs); obs,r,term,trunc,info=env.step(a)
    if i%25==0 or term: print(i,np.round(rb(obs),3),"phase",ap.phase,"stuck",ap.stuck, ch(obs))
    if term: break
