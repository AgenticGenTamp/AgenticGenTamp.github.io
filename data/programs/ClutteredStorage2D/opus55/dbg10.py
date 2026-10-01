import sys
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); oc=int(sys.argv[2]) if len(sys.argv)>2 else None
env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':oc} if oc else None)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
prev=None;t0=0
for t in range(1000):
    a=ap.get_action(obs)
    if ap.task!=prev:
        print(t, ap.task, round(ap.rx,2), round(ap.ry,2), 'lazy', sorted(ap.lazy_cols), 'tot', getattr(ap,'plan_total',None))
        prev=ap.task
    obs,r,term,trunc,info=env.step(a)
    if term: print('done',t); break
