import numpy as np
from env_client import make_env


def one(seed, actions):
    env = make_env()
    obs, info = env.reset(seed=seed)
    print("RESET", seed, np.round(obs, 4).tolist(), info)
    for name, act in actions:
        nxt, rew, term, trunc, inf = env.step(np.asarray(act, dtype=np.float32))
        print(name, "delta", np.round(nxt-obs, 4).tolist(), "state", np.round(nxt, 4).tolist(), rew, term, trunc, inf)
        obs = nxt
    env.close()


if __name__ == "__main__":
    z = np.zeros(11)
    acts = [("zero", z), ("bx", np.eye(11)[0]*.4), ("by", np.eye(11)[1]*.4),
            ("rot", np.eye(11)[2]*.4), ("j1", np.eye(11)[3]*.4)]
    one(0, acts)
    for s in range(1, 5):
        one(s, [])
