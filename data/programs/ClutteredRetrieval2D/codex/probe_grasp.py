"""Drive the arm toward the target and log attachment behavior."""

import math
import sys
import numpy as np
from env_client import make_env

env = make_env()
s, info = env.reset(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
types = {t.name: t for t in env.observation_space.types}


def one(n): return list(s.get_objects(types[n]))[0]
def v(o, n): return float(s.get(o, n))
def step(a):
    global s
    s, r, done, trunc, info = env.step(np.asarray(a, dtype=np.float32))
    return done


for i in range(80):
    ro, bl = one("crv_robot"), s.get_object_from_name("obstruction5")
    desired = math.atan2(v(bl, "y") - v(ro, "y"), v(bl, "x") - v(ro, "x")) - math.pi
    err = (desired - v(ro, "theta") + math.pi) % (2*math.pi) - math.pi
    if abs(err) < .01: break
    step([0, 0, max(-.196, min(.196, err)), 0, 0])
print("oriented", i, [v(one("crv_robot"), q) for q in ("x","y","theta","arm_joint","arm_length")])
for i in range(40):
    ro, bl = one("crv_robot"), s.get_object_from_name("obstruction5")
    print(i, "robot",round(v(ro,"x"),3),round(v(ro,"y"),3), "joint", round(v(ro,"arm_joint"),3), "vac",v(ro,"vacuum"),
          "block",round(v(bl,"x"),3),round(v(bl,"y"),3))
    dist = math.hypot(v(bl,"x")-v(ro,"x"), v(bl,"y")-v(ro,"y"))
    move = .05 if dist > .05 else 0.0
    step([-move*math.cos(v(ro,"theta")), -move*math.sin(v(ro,"theta")), 0, .1, 1])
print("retract")
for i in range(8):
    ro = one("crv_robot")
    rects = list(s.get_objects(types["rectangle"]))
    print(i, "robot", round(v(ro,"x"),3), round(v(ro,"y"),3), "joint",round(v(ro,"arm_joint"),3),
          [(o.name, round(v(o,"x"),3), round(v(o,"y"),3)) for o in rects])
    step([0, 0, 0, -.1, 1])
env.close()
