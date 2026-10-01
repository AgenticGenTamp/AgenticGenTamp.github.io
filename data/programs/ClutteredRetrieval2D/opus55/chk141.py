import sys, numpy as np
from env_client import make_env
import approach
from approach import *
seed=141; upto=37
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for step in range(upto):
    a=ap.get_action(obs); obs, *_ = env.step(a)
ap._parse(obs)
print('q', ap.q, 'held', ap.held)
obs_all, names = ap.make_obstacles()
model = Model(obs_all, None, margin=0.0)
q=ap.q.copy()
print('cur per_obj m0', [names[i] for i in np.nonzero(model.per_obj(q[None],0.0)[0])[0]])
for m in [0.0,0.002,0.005,0.01]:
    print(m,[names[i] for i in np.nonzero(model.per_obj(q[None],m)[0])[0]])
# try unit moves in env
for d in [(0.01,0,0,0),(-0.01,0,0,0),(0,0.01,0,0),(0,-0.01,0,0),(0,0,0.02,0),(0,0,-0.02,0),(0,0,0,0.01),(0,0,0,-0.01)]:
    e2 = make_env(); o2,_ = e2.reset(seed=seed)
    ap2 = GeneratedApproach(e2.action_space, e2.observation_space, {}); ap2.reset(o2, info)
    for step in range(upto):
        o2,*_ = e2.step(ap2.get_action(o2))
    act=np.array(list(d)+[0.0],dtype=np.float32)
    o3,*_=e2.step(act); ap2._parse(o3); print(d, 'moved', not np.allclose(ap2.q, q, atol=1e-4))
    e2.close()
for n in ap.rects: print(n, np.round(ap.rects[n],3).tolist() if hasattr(ap.rects[n],'__len__') else ap.rects[n])
