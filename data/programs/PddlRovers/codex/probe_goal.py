import sys
import numpy as np
from env_client import make_env


def rows(s, typ, feats):
    out = []
    t = next(t for t in env.observation_space.types if t.name == typ)
    for o in s.get_objects(t):
        out.append((o.name,) + tuple(round(float(s.get(o, f)), 3) for f in feats))
    return out


env = make_env()
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
s, info = env.reset(seed=seed)
print("seed", seed, "max_steps", env.max_steps, "reset_info", info)
print("rovers", rows(s, "rover", ["x", "y", "theta", "store_full", "calibrated", "at_home"]))
print("lander", rows(s, "lander", ["x", "y", "z"]))
print("objectives", rows(s, "objective", ["x", "y", "z", "have_image_rover0", "have_image_rover1", "received_image"]))
print("samples", rows(s, "sample", ["x", "y", "z", "is_soil", "analyzed_rover0", "analyzed_rover1", "received_analysis"]))
print("obstacles", rows(s, "obstacle", ["x", "y", "z", "half_x", "half_y", "half_z"]))
for label, a in [
    ("zero", np.zeros(8, np.float32)),
    ("sample", np.array([0,0,0,-5/6,0,0,0,-5/6], np.float32)),
    ("cal", np.array([0,0,0,-.5,0,0,0,-.5], np.float32)),
    ("image", np.array([0,0,0,-1/6,0,0,0,-1/6], np.float32)),
    ("send", np.array([0,0,0,.5,0,0,0,.5], np.float32)),
    ("drop", np.array([0,0,0,5/6,0,0,0,5/6], np.float32)),
]:
    s, r, term, trunc, info = env.step(a)
    print(label, r, term, trunc, info, rows(s, "rover", ["x","y","store_full","calibrated","at_home"]))
env.close()
