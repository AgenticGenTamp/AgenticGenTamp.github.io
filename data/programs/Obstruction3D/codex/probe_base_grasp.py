from env_client import make_env
import numpy as np


def val(s, name, feature):
    return float(s.get(s.get_object_from_name(name), feature))


def trial(dx, dy, joints=None):
    env = make_env(); s, info = env.reset(seed=0)
    a = np.zeros(11, np.float32); a[0] = dx; a[1] = dy
    if joints is not None: a[3:10] = joints
    s, _, _, _, _ = env.step(a)
    a[:] = 0; a[10] = -1
    s, _, _, _, _ = env.step(a)
    hit = [n for n in s.get_object_names() if n != "robot" and val(s,n,"grasp_active") > .5]
    out=(val(s,"robot","pos_base_x"),val(s,"robot","pos_base_y"),val(s,"robot","finger_state"),val(s,"robot","grasp_active"),hit)
    env.close(); return out


for dx in [-.2,-.1,0,.1,.2]:
    for dy in [-.2,-.1,0,.1,.2]:
        print(dx,dy,trial(dx,dy))
