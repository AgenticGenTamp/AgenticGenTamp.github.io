from env_client import make_env
import numpy as np

for seed in range(12):
    env = make_env(); s, info = env.reset(seed=seed)
    rob = s.get_object_from_name("robot")
    start = tuple(float(s.get(rob, f)) for f in ("pos_base_x", "pos_base_y"))
    last = (None, None, None)
    for t in range(160):
        a = np.zeros(11, np.float32)
        a[0] = .1; a[1] = .1
        s, r, term, trunc, inf = env.step(a)
        rob = s.get_object_from_name("robot")
        last = (float(s.get(rob,"pos_base_x")), float(s.get(rob,"pos_base_y")), r)
        if term or trunc or r != -1.0: break
    print(seed, info["object_count"], "start", tuple(round(x,2) for x in start),
          "end", tuple(round(x,2) for x in last), "t", t+1, "done", term, trunc)
    env.close()
