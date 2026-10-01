import numpy as np
from env_client import make_env
from approach import GeneratedApproach
env = make_env(); obs, info = env.reset(seed=0)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
hist=[]
for t in range(1000):
    a = ap.get_action(obs); obs, r, term, trunc, info = env.step(a)
    names=[n for n in obs.get_object_names() if n.startswith('small')]
    P=[(obs.get(obs.get_object_from_name(n),'x'),obs.get(obs.get_object_from_name(n),'y'),obs.get(obs.get_object_from_name(n),'vx'),obs.get(obs.get_object_from_name(n),'vy')) for n in names]
    hist.append(P)
    if t in (540,560,600,850,870,885) or term:
        xs=np.array([p[0] for p in P]); ys=np.array([p[1] for p in P])
        print(t, term, 'x>1.8:',(xs>1.8).sum(), 'x>1.8&y<1.5:',((xs>1.8)&(ys<1.5)).sum(), 'x>1.8&y<0.3', ((xs>1.8)&(ys<0.3)).sum(), 'x>2:',(xs>2.0).sum())
    if term: break
for t in range(len(hist)-8, len(hist)):
    P=hist[t]
    print(t, sorted([round(p[1],2) for p in P if p[0]>1.8]))
