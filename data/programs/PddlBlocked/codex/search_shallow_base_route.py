"""Probe post-blocker green grasps over base heading and outside-table route."""
import math
import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach


def g(s, n, f):
    return float(s.get(s.get_object_from_name(n), f))


def drive(env, s, p, target, q, limit=40):
    for _ in range(limit):
        a = np.zeros(11, np.float32)
        a[:2] = np.clip(target-p.robot(s), -.2, .2)
        a[2] = np.clip(p.w(p.theta-g(s, "robot", "base_rot")), -.2, .2)
        for j in range(7):
            d = q[j]-g(s, "robot", "joint_"+str(j+1))
            if j in (4, 6): d = p.w(d)
            a[3+j] = np.clip(d, -.2, .2)
        a[10] = 1.
        s, *_ = env.step(a)
    return s


def setup(env, seed):
    s, info = env.reset(seed=seed)
    p = GeneratedApproach(env.action_space, env.observation_space, {})
    p.reset(s, info)
    for _ in range(80):
        if p.stage == 5: break
        s, *_ = env.step(p.get_action(s))
    return s, p


seed = int(sys.argv[1])
short = len(sys.argv) > 2 and sys.argv[2] == "short"
aligned = len(sys.argv) > 2 and sys.argv[2] == "aligned"
SHORT_Q = np.array([.75099605, .64684612, 1.00101125, -2.32130003,
                    -2.48106384, -2.09400010, -2.10809493])
SHORT_OFF = np.array([.49231529, .40115744])
env = make_env()
try:
    if aligned:
        _, probe = setup(env, seed)
        headings = [probe.w(math.atan2(-probe.out[1],-probe.out[0])-probe.TOOL_YAW)]
    else:
        headings = np.arange(-math.pi, math.pi, .10)
    for theta in headings:
        s, p = setup(env, seed)
        p.theta = float(theta)
        c, z = math.cos(theta), math.sin(theta)
        off = SHORT_OFF if (short or aligned) else p.OFF
        q = SHORT_Q if (short or aligned) else p.Q
        p.off = np.array([c*off[0]-z*off[1], z*off[0]+c*off[1]])
        target = p.target(p.green)
        if aligned: target[0] = min(5., target[0])
        if not (-1. <= target[0] <= 5. and -1. <= target[1] <= 1.75):
            continue
        qlift = q.copy(); qlift[1] -= .2
        # Route around the near-table perimeter on the same north/south side.
        side = 1.65 if target[1] >= 0 else -1.
        for waypoint in (np.array([3.35, side]), np.array([target[0], side]), target):
            s = drive(env, s, p, waypoint, qlift, 18)
        s = drive(env, s, p, target, q, 18)
        a = np.zeros(11, np.float32); a[10] = -1.
        s, *_ = env.step(a)
        if g(s, "green0", "grasp_active") > .5:
            print("HIT", seed, round(theta, 4), "target", target, "base", p.robot(s))
            break
    else:
        print("MISS", seed)
finally:
    env.close()
