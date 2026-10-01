import math
import numpy as np
from env_client import make_env


def val(s, name, f):
    return float(s.get(s.get_object_from_name(name), f))


def clip(x, m):
    return max(-m, min(m, x))


def angle(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def run(seed, yoff, arm_goal):
    e = make_env(); s, _ = e.reset(seed=seed)
    hx, hy = val(s, "hook", "x"), val(s, "hook", "y")
    target = (hx, hy + yoff, -math.pi / 2)
    for t in range(180):
        rx, ry, rt = val(s, "robot", "x"), val(s, "robot", "y"), val(s, "robot", "theta")
        if t < 130:
            a = [clip(target[0]-rx,.03), clip(target[1]-ry,.03), clip(angle(target[2]-rt),.098), clip(arm_goal-val(s,"robot","arm_length"),.08), .015]
        else:
            a = [0, 0, 0, 0, -.015]
        s, r, term, trunc, _ = e.step(np.array(a))
        if t in (100,129,140,160,179) or val(s,"hook","held"):
            print(seed,yoff,arm_goal,t,"robot",*[round(val(s,"robot",f),3) for f in ("x","y","theta","arm_length","finger_gap")],"hook",*[round(val(s,"hook",f),3) for f in ("x","y","theta","held")])
        if val(s,"hook","held"):
            break
    e.close()


for yo in (.65,.75,.85, .95):
    run(0, yo, .4)
