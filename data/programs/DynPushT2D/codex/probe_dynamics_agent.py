"""Compact black-box probes for DynPushT2D; exploration only."""
import math
import numpy as np
from env_client import make_env


def show_seed(seed):
    env = make_env()
    obs, info = env.reset(seed=seed)
    vals = obs
    print("seed", seed, "max", env.max_steps,
          "block", np.round(vals[[0,1,2]], 4).tolist(),
          "robot", np.round(vals[[16,17,28]], 4).tolist(),
          "goal", np.round(vals[[29,30,31]], 4).tolist(),
          "dims", np.round(vals[[12,13,14,15]], 4).tolist(),
          "bounds", np.round(env.observation_space.low[[0,1,16,17]],2).tolist(),
          np.round(env.observation_space.high[[0,1,16,17]],2).tolist())
    env.close()


def rollout(seed, actions, label):
    env = make_env(); o, _ = env.reset(seed=seed)
    initial = o.copy(); last = o
    for i, a in enumerate(actions):
        last, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
        if term or trunc:
            break
    print(label, "n", i+1,
          "robot d", np.round(last[16:18]-initial[16:18],4).tolist(),
          "block dxyth", np.round(last[:3]-initial[:3],4).tolist(),
          "final robot", np.round(last[16:18],4).tolist(),
          "v", np.round(last[3:6],4).tolist(), "term",term,"trunc",trunc)
    env.close()


for s in range(5):
    show_seed(s)

rollout(0, [[0.01, -0.02]], "free-one")
rollout(0, [[0.049, 0]]*30, "right-bound/contact")
rollout(0, [[-0.049, 0]]*30, "left-bound/contact")
rollout(0, [[0, 0.049]]*30, "up-bound/contact")
rollout(0, [[0, -0.049]]*30, "down-bound/contact")

# Seed 0's object is high/left, leaving three direct clear rays and a clear
# upper ray after shifting to the far-right corridor.
rollout(0, [[-0.049, 0]]*120, "left-wall")
rollout(0, [[0.049, 0]]*120, "right-wall")
rollout(0, [[0, -0.049]]*120, "bottom-wall")

env=make_env(); o,_=env.reset(seed=0)
o=move_obs=None
for _ in range(60):
    o,*_=env.step(np.array([.049,0],np.float32))
for _ in range(120):
    o,*_=env.step(np.array([0,.049],np.float32))
print("top-wall-via-right final robot",np.round(o[16:18],4).tolist())
env.close()
