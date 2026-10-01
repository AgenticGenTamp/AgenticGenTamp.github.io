"""Map the top-wall opening by extending an empty vertical arm."""
import math
import numpy as np
from env_client import make_env


for gx in np.linspace(0.05, 4.95, 50):
    env = make_env(); s, _ = env.reset(seed=11, options={"object_count": 1})
    r = s.get_object_from_name("robot")
    for _ in range(100):
        x, y, t, a = (float(s.get(r, q)) for q in ("x", "y", "theta", "arm_joint"))
        et = (math.pi/2-t+math.pi)%(2*math.pi)-math.pi
        act = np.array([np.clip(gx-x,-.05,.05), np.clip(2.0-y,-.05,.05),
                        np.clip(et,-.196,.196), np.clip(.2-a,-.1,.1), 0], np.float32)
        s, *_ = env.step(act)
        if max(abs(gx-float(s.get(r,"x"))), abs(2-float(s.get(r,"y"))), abs(et)) < .004: break
    for _ in range(9):
        s, *_ = env.step(np.array([0,0,0,.1,0], np.float32))
    arm = float(s.get(r,"arm_joint"))
    print(f"{gx:.3f} {arm:.3f}")
    env.close()
