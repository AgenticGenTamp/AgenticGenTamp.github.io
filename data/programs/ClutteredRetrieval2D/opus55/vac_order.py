import numpy as np
from env_client import make_env
from approach import *
env = make_env(); obs, info = env.reset(seed=11)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
path = ap.plan_grasp('target_block')
ap.enqueue_path(path, 0.0)
acts = [q[0] for q in ap.queue]
for d in acts[:-1]:
    obs, *_ = env.step(np.array([*d, 0], dtype=np.float32))
# last move with vac=1
obs, *_ = env.step(np.array([*acts[-1], 1], dtype=np.float32))
ap._parse(obs); b0 = ap.center('target_block')
obs, *_ = env.step(np.array([-0.03, 0, 0, 0, 1], dtype=np.float32))
ap._parse(obs); b1 = ap.center('target_block')
print("grasp on final move step: block moved", b1 - b0)
# now release combined with move
obs, *_ = env.step(np.array([-0.03, 0, 0, 0, 0], dtype=np.float32))
ap._parse(obs); b2 = ap.center('target_block')
print("release+move step: block moved", b2 - b1)
