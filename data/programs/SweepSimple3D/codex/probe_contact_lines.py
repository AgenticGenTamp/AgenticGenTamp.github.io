"""Locate arm/tool contact by driving straight through the wiper on parallel lines."""

import math
import numpy as np
from env_client import make_env


def val(s, n, f):
    return float(s.get(s.get_object_from_name(n), f))


def xy(s, n):
    if n == "robot":
        return np.array([val(s, n, "pos_base_x"), val(s, n, "pos_base_y")])
    return np.array([val(s, n, "x"), val(s, n, "y")])


def scan(env, xoff, q2off, grip, q4off=0):
    s, _ = env.reset(seed=0, options={"object_count": 1})
    w0 = xy(s, "wiper_0")
    q20 = val(s, "robot", "pos_arm_joint2")
    q40 = val(s, "robot", "pos_arm_joint4")
    target = w0 + np.array([xoff, .90])
    # Reach an exact, common starting pose.
    for _ in range(15):
        a = np.zeros(11, np.float32)
        a[:2] = np.clip(target - xy(s, "robot"), -.1, .1)
        yawerr = (-math.pi / 2 - val(s, "robot", "pos_base_rot") + math.pi) % (2*math.pi)-math.pi
        a[2] = np.clip(yawerr, -.1, .1)
        a[4] = np.clip(q20 + q2off - val(s, "robot", "pos_arm_joint2"), -.1, .1)
        a[6] = np.clip(q40 + q4off - val(s, "robot", "pos_arm_joint4"), -.1, .1)
        a[10] = grip
        s, _, _, _, _ = env.step(a)
    first = None
    for k in range(24):
        a = np.zeros(11, np.float32); a[1] = -.04; a[10] = grip
        s, _, _, _, _ = env.step(a)
        delta = xy(s, "wiper_0") - w0
        if np.linalg.norm(delta) > .002 and first is None:
            first = (k, xy(s, "robot").copy(), delta.copy())
            break
    return first


if __name__ == "__main__":
    env = make_env()
    for q2off in (-1., 0., 1.):
      for q4off in (-1., 0., 1.):
        for xoff in (-.45, -.25, -.05, .15, .35, .55):
            hit = scan(env, xoff, q2off, 0, q4off)
            # Contact while base is >35 cm from origin implies arm contact.
            print("q2/q4/x", q2off, q4off, xoff, "contact", hit, flush=True)
    env.close()
