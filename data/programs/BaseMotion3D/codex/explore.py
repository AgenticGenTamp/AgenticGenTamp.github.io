import numpy as np
from env_client import make_env


def show(seed, actions):
    env = make_env()
    obs, info = env.reset(seed=seed)
    print("seed", seed, "max", env.max_steps, "initial", np.round(obs, 4), "info", info)
    for label, action in actions:
        old = obs.copy()
        obs, rew, term, trunc, info = env.step(np.asarray(action, dtype=np.float32))
        print(label, "delta", np.round(obs-old, 4), "obs", np.round(obs,4), rew, term, trunc, info)
        if term or trunc: break
    env.close()


if __name__ == "__main__":
    z = np.zeros(11, dtype=np.float32)
    acts = [("zero", z), ("x+", np.array([.4]+[0]*10)),
            ("y+", np.array([0,.4]+[0]*9)),
            ("rot+", np.array([0,0,.4]+[0]*8))]
    for s in range(3): show(s, acts)
