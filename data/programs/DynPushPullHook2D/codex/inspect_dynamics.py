from env_client import make_env
import numpy as np


def vals(s, name):
    o = s.get_object_from_name(name)
    fs = ("x", "y", "theta", "vx_base", "vy_base") if name == "robot" else ("x", "y", "theta", "vx", "vy", "held")
    return tuple(round(float(s.get(o, f)), 3) for f in fs)


for mode, action in (("zero", np.zeros(5)), ("up", np.array([0, .05, 0, 0, 0])),
                     ("arm", np.array([0, 0, 0, .1, 0])), ("close", np.array([0, 0, 0, 0, -.02]))):
    e = make_env(); s, _ = e.reset(seed=0)
    print("MODE", mode, "initial", vals(s, "robot"), vals(s, "hook"), vals(s, "target_block"))
    for t in range(1, 101):
        s, r, term, trunc, info = e.step(np.asarray(action, dtype=e.action_space.dtype))
        if t in (1, 5, 10, 20, 40, 60, 100) or term:
            print(t, vals(s, "robot"), vals(s, "hook"), vals(s, "target_block"), r, term, trunc)
        if term or trunc: break
    e.close()
