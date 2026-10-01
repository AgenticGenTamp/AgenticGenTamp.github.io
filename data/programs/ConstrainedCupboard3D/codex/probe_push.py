import numpy as np
from env_client import make_env


def get(s, name, f):
    return float(s.get(s.get_object_from_name(name), f))


e = make_env()
s, info = e.reset(seed=1)
rod = "cuboid_0"
grip = 1.0
for step in range(80):
    bx, by = get(s, "robot", "pos_base_x"), get(s, "robot", "pos_base_y")
    ox, oy = get(s, rod, "x"), get(s, rod, "y")
    a = np.zeros(11, np.float32); a[-1] = grip
    if step < 8:
        a[1] = np.clip((oy - by) * 1.5, -.1, .1)
    elif step < 18:
        a[0] = np.clip((ox - .38 - bx) * 1.5, -.1, .1)
    else:
        a[0] = .1
    s, rew, term, trunc, inf = e.step(a)
    if step % 4 == 0 or term:
        qz=get(s,rod,"qz"); qw=get(s,rod,"qw")
        print(step, "base", round(get(s,"robot","pos_base_x"),3),round(get(s,"robot","pos_base_y"),3),
              "rod",round(get(s,rod,"x"),3),round(get(s,rod,"y"),3),round(qw,3),round(qz,3),"r",rew,term)
    if term: break
e.close()
