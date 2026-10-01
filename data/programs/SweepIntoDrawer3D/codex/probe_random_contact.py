import numpy as np
from env_client import make_env


def run(seed, mode):
    rng = np.random.default_rng(seed + 1000 * mode)
    env = make_env()
    o0, _ = env.reset(seed=seed)
    o = o0.copy()
    best = 0.0
    for t in range(300):
        a = np.zeros(11, dtype=np.float32)
        if mode == 0:
            a[3:10] = rng.choice([-0.1, 0.0, 0.1], 7)
        elif mode == 1:
            # Hold a direction for long enough to yield substantial motion.
            rng2 = np.random.default_rng(seed + 1000 * mode + t // 20)
            a[3:10] = rng2.choice([-0.1, 0.0, 0.1], 7)
        else:
            a[3:10] = 0.1 * np.sin(0.07*t + np.arange(7))
        a[10] = 0.0 if (t // 50) % 2 == 0 else 1.0
        prev = o.copy()
        o, r, term, trunc, info = env.step(a)
        delta_w = np.linalg.norm(o[147:150] - o0[147:150])
        delta_c = np.max(np.linalg.norm(o[:80].reshape(5,16)[:,:3] - o0[:80].reshape(5,16)[:,:3], axis=1))
        delta_d = np.max(np.abs(o[103:109] - o0[103:109]))
        score = delta_w + delta_c + delta_d
        if score > best + .002 or r != -1.0:
            best = score
            print("seed/mode/t", seed, mode, t, "r", r,
                  "w", np.round(o[147:150],3).tolist(), "dw", round(float(delta_w),3),
                  "dc", round(float(delta_c),3), "draw", np.round(o[103:109],3).tolist(),
                  "j", np.round(o[128:136],2).tolist(), "a", np.round(a,2).tolist())
        if term or trunc:
            print("END", t, term, trunc, info)
            break
    env.close()


for mode in range(3):
    run(0, mode)
