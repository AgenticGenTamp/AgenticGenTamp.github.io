"""Focused experiments for due-east, no-spare blocker cases."""
import math
import sys

import numpy as np

from env_client import make_env


SEEDS = (34, 65, 101, 108, 126, 142, 147, 149, 191, 195)
Q = np.array([.75099605, .64684612, 1.00101125, -2.32130003,
              -2.48106384, -2.09400010, -2.10809493])
OFF = np.array([.49231529, .40115744])


def g(s, n, f):
    return float(s.get(s.get_object_from_name(n), f))


def pos(s, n):
    return np.array([g(s, n, "pose_x"), g(s, n, "pose_y")])


def wrap(x):
    return (x+math.pi) % (2*math.pi)-math.pi


def move(env, s, target, theta, q=Q, grip=1., steps=40):
    for _ in range(steps):
        a = np.zeros(11, np.float32)
        here = np.array([g(s, "robot", "base_x"), g(s, "robot", "base_y")])
        a[:2] = np.clip(target-here, -.2, .2)
        a[2] = np.clip(wrap(theta-g(s, "robot", "base_rot")), -.2, .2)
        if q is not None:
            for j in range(7):
                dq = q[j]-g(s, "robot", "joint_"+str(j+1))
                if j in (4, 6):
                    dq = wrap(dq)
                a[3+j] = np.clip(dq, -.2, .2)
        a[10] = grip
        s, *_ = env.step(a)
    return s


def grasp_test(seed, theta):
    env = make_env(); s, _ = env.reset(seed=seed); blocker = pos(s, "blocker")
    c, z = math.cos(theta), math.sin(theta)
    target = blocker-np.array([c*OFF[0]-z*OFF[1], z*OFF[0]+c*OFF[1]])
    side = -1.6 if target[1] < .8 else 1.6
    waypoints = (np.array([3.4, side]), np.array([target[0], side]), target)
    for wi, waypoint in enumerate(waypoints):
        s = move(env, s, waypoint, theta, q=Q if wi == 2 else None)
    a = np.zeros(11, np.float32); a[10] = -1.; s, *_ = env.step(a)
    held = g(s, "blocker", "grasp_active") > .5
    print("GRASP", seed, round(theta, 4), held, "target", target.round(4),
          "actual", [round(g(s, "robot", x), 4) for x in ("base_x", "base_y")])
    env.close()


def main():
    env = make_env()
    for seed in SEEDS if len(sys.argv) == 1 else map(int, sys.argv[1:]):
        s, _ = env.reset(seed=seed)
        green, blocker = pos(s, "green0"), pos(s, "blocker")
        d = (blocker-green) / np.linalg.norm(blocker-green)
        print(seed, "green", green.round(5), "block", blocker.round(5),
              "d", d.round(5), "angle", round(math.atan2(d[1], d[0]), 5))
    env.close()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "grasp":
        grasp_test(int(sys.argv[2]), float(sys.argv[3]))
    else:
        main()
