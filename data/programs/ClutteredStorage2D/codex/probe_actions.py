import numpy as np
from env_client import make_env


def vals(s):
    r = s.get_object_from_name("robot")
    bs = [s.get_object_from_name(n) for n in s.get_object_names() if n.startswith("block")]
    return ([round(float(s.get(r, f)), 4) for f in ("x", "y", "theta", "arm_joint", "vacuum")],
            [(b.name, round(float(s.get(b, "x")), 4), round(float(s.get(b, "y")), 4)) for b in bs])


env = make_env()
s, info = env.reset(seed=1)
print("init", vals(s))
for a in ([.05, 0, 0, 0, 0], [0, .05, 0, 0, 0], [0, 0, .1963495, 0, 0],
          [0, 0, 0, .1, 0], [0, 0, 0, -.1, 0]):
    s, rew, done, trunc, info = env.step(np.array(a, dtype=np.float32))
    print(a, vals(s), rew, done, trunc)
env.close()
