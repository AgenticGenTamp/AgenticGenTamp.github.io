"""Coarse chassis/wiper impact search; intentionally separate from policy."""
import math
import sys
import numpy as np
from env_client import make_env


def val(s, name, feature):
    return float(s.get(s.get_object_from_name(name), feature))


def xy(s, name):
    if name == "robot":
        return np.array([val(s, name, "pos_base_x"), val(s, name, "pos_base_y")])
    return np.array([val(s, name, "x"), val(s, name, "y")])


def run(angle_deg, lateral, speed, push_steps=35, object_count=1, seed=0):
    env = make_env()
    s, _ = env.reset(seed=seed, options={"object_count": object_count})
    cubes = sorted(n for n in s.get_object_names() if n.startswith("cube_"))
    w0 = xy(s, "wiper_0")
    c0s = {n: xy(s, n) for n in cubes}
    center = np.mean(list(c0s.values()), axis=0)
    direct = center - w0
    direct /= np.linalg.norm(direct)
    th = math.radians(angle_deg)
    direction = np.array([direct[0]*math.cos(th)-direct[1]*math.sin(th),
                          direct[0]*math.sin(th)+direct[1]*math.cos(th)])
    side = np.array([-direction[1], direction[0]])
    start = w0 - 0.48*direction + lateral*side
    # Position with small closed-loop actions, stopping before contact.
    for _ in range(24):
        a = np.zeros(11, np.float32)
        a[:2] = np.clip(0.8*(start-xy(s, "robot")), -.1, .1)
        s, _, _, _, _ = env.step(a)
    maxcd = 0.0
    minz = 9.0
    reward_peak = -1.0
    contact = None
    for k in range(push_steps):
        a = np.zeros(11, np.float32)
        a[:2] = speed*direction
        s, reward, done, trunc, _ = env.step(a)
        wd = float(np.linalg.norm(xy(s, "wiper_0")-w0))
        cd = max(float(np.linalg.norm(xy(s, n)-c0s[n])) for n in cubes)
        maxcd = max(maxcd, cd)
        minz = min(minz, val(s, "wiper_0", "z"))
        reward_peak = max(reward_peak, reward)
        if contact is None and wd > .003:
            contact = k
        if done or trunc:
            break
    qz = abs(val(s, "wiper_0", "qz"))
    qw = abs(val(s, "wiper_0", "qw"))
    tilt = 2*math.asin(min(1.0, math.hypot(val(s,"wiper_0","qx"), val(s,"wiper_0","qy"))))
    result = (angle_deg, lateral, speed, contact, maxcd,
              *(xy(s,"wiper_0")-w0), val(s,"wiper_0","z"), tilt,
              *(xy(s,cubes[0])-c0s[cubes[0]]), reward_peak, bool(done))
    env.close()
    print("ang lat sp hit maxcd wdx wdy wz tilt cdx cdy r done", *
          [round(x,4) if isinstance(x, float) else x for x in result], flush=True)
    if object_count > 1:
        print("cube_deltas", {n: np.round(xy(s, n)-c0s[n], 4).tolist()
                              for n in cubes}, flush=True)


if __name__ == "__main__":
    # Optional single configuration for refinement.
    if len(sys.argv) > 1:
        run(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]),
            int(sys.argv[4]) if len(sys.argv) > 4 else 35,
            int(sys.argv[5]) if len(sys.argv) > 5 else 1,
            int(sys.argv[6]) if len(sys.argv) > 6 else 0)
    else:
        for angle in (-30, -15, 0, 15, 30):
            for lateral in (-.18, 0., .18):
                run(angle, lateral, .035)
