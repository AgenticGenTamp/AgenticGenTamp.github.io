import numpy as np
from env_client import make_env


env = make_env()
o, _ = env.reset(seed=0)
print("start base", o[125:128], "draw", o[103:109], "wiper", o[147:150])

# Move from the east end to the negative-y long face and turn north.
for t in range(16):
    a = np.zeros(11, np.float32)
    a[:3] = [-0.035, -0.048, -0.10]
    a[10] = 0.0
    o, r, term, trunc, _ = env.step(a)
    if t in (7, 15):
        print("nav", t+1, "base", o[125:128].round(3), "draw",o[103:109].round(3))

# Advance the home-pose gripper toward the face, close, then pull away.
for phase, count, vy, grip in [("advance", 3, .08, 0.0), ("close", 3, 0.0, 1.0), ("pull", 6, -.08, 1.0)]:
    for t in range(count):
        a = np.zeros(11, np.float32)
        a[1] = vy
        a[10] = grip
        o, r, term, trunc, _ = env.step(a)
        print(phase, t+1, "base",o[125:128].round(3),"grip",round(float(o[135]),3),"draw",o[103:109].round(3),"wiper",o[147:150].round(3),"r",r)
env.close()
