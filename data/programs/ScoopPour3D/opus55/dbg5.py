import sys, numpy as np, kin
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); T=int(sys.argv[2])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
prev=None
for t in range(T):
    a=ap.get_action(obs)
    obs,r,term,trunc,info=env.step(a)
    if ap.phase!=prev or t%10==0:
        base,q=ap._robot(obs); p,_=kin.fk_world(base,q)
        c=ap._objs(obs)[0].get(ap.cur) if ap.cur else None
        print(t, ap.phase, ap.pt, ap.cur, 'tool',p.round(3), 'cube', None if c is None else c[:3].round(3), 'err',round(getattr(ap,'tool_err',0),3),'I',np.abs(ap.I).max().round(3), 'base',base.round(3), 'yaw', round(getattr(ap,'yaw',0),2))
        prev=ap.phase
