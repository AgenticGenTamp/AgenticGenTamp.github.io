import numpy as np
from env_client import make_env


def run(joint, sign, steps=7):
    env = make_env()
    obs, _ = env.reset(seed=0)
    init = obs.copy()
    best = np.zeros(6)
    for t in range(steps):
        a = np.zeros(11, np.float32)
        a[3 + joint] = sign * .1
        a[10] = 0.0
        obs, rew, term, trunc, info = env.step(a)
        best = np.maximum(best, np.abs(obs[103:109]))
    cube_shift = np.linalg.norm(obs[np.array([[16*i,16*i+1,16*i+2] for i in range(5)])] - init[np.array([[16*i,16*i+1,16*i+2] for i in range(5)])], axis=1)
    print("joint", joint+1, "sign", sign, "q", obs[128:135].round(3).tolist(), "draw",obs[103:109].round(4).tolist(),"best",best.round(4).tolist(),"cshift",cube_shift.round(3).tolist(),"wiper",obs[147:150].round(3).tolist(),"reward",rew)
    env.close()


for j in [1, 3, 5, 0, 2, 4, 6]:
    for s in [-1, 1]:
        run(j, s)
