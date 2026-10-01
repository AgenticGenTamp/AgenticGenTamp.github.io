import numpy as np
from env_client import make_env


def run(seed, mode="direct", verbose=False, env=None):
    own_env = env is None
    if own_env:
        env = make_env()
    obs, _ = env.reset(seed=seed)
    target = obs[19:21].copy()
    for step in range(1000):
        a = np.zeros(11, dtype=np.float32)
        delta = target - obs[:2]
        if mode == "direct":
            a[:2] = np.clip(delta, -.4, .4)
        elif mode == "unit":
            norm = np.linalg.norm(delta)
            a[:2] = delta * min(.4 / max(norm, 1e-9), 1.0)
        elif mode == "xtheny":
            if abs(delta[0]) > .05: a[0] = np.clip(delta[0], -.4, .4)
            else: a[1] = np.clip(delta[1], -.4, .4)
        obs, rew, term, trunc, info = env.step(a)
        if term or trunc:
            if own_env: env.close()
            if verbose: print(seed, target, step+1, term, obs[:3], info)
            return step+1, term, target, obs
    if own_env: env.close()
    return 1000, False, target, obs


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv)>1 else "direct"
    out=[]
    env = make_env()
    n = int(sys.argv[2]) if len(sys.argv)>2 else 100
    start = int(sys.argv[3]) if len(sys.argv)>3 else 0
    for seed in range(start, start+n):
        r=run(seed, mode, False, env)
        out.append(r[:2])
    env.close()
    print(mode, "success", sum(x[1] for x in out), "mean", np.mean([x[0] for x in out]), "max", max(x[0] for x in out))
