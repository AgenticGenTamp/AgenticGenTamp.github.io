import numpy as np, approach
from env_client import make_env
from approach import *
env = make_env()
rng = np.random.default_rng(1)
data = []
for seed in range(6):
    obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
    tc = ap.center('target_block')
    for k in range(400):
        ap._parse(obs); q = ap.q.copy()
        d = tc - q[:2]
        a = np.array([*(np.clip(d*0.2,-0.05,0.05) + rng.uniform(-0.03,0.03,2)), rng.uniform(-0.19,0.19), rng.uniform(-0.1,0.1), 0.0])
        a[:2] = np.clip(a[:2], -0.05, 0.05)
        obs2, *_ = env.step(a.astype(np.float32))
        ap2 = ap; rects = dict(ap.rects)
        ap._parse(obs2)
        acc = np.abs(ap.q - q).max() > 1e-9
        qn = q + a[:4]; qn[2] = wrap(qn[2]); qn[3] = np.clip(qn[3], 0.1, 0.2)
        data.append((rects, qn, acc))
        obs = obs2
print("n", len(data), "rejected", sum(not d[2] for d in data))
for w in [0.0, 0.002, 0.005, 0.01, 0.015, 0.02]:
    approach.ARM_W = w
    err = 0; fp = 0
    for rects, qn, acc in data:
        polys = [rect_corners(*v) for n, v in rects.items() if n != 'target_region']
        m = Model(Obstacles(polys), None, margin=0.0)
        pred = m.free(qn)
        if pred != acc: err += 1; fp += (pred and not acc)
    print("ARM_W", w, "errors", err, "pred-free-but-rejected", fp)
