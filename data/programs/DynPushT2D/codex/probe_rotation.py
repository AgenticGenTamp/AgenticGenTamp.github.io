"""Targeted black-box probes for DynPushT2D rotational mechanics."""

import argparse
import math

import numpy as np

from env_client import make_env


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def resets(seeds):
    env = make_env()
    print("seed block(x,y,th) robot(x,y) goal(x,y,th) dims(w,lh,lv,m)")
    for seed in seeds:
        s, _ = env.reset(seed=seed)
        print(seed, np.round(s[[0, 1, 2, 16, 17, 29, 30, 31, 12, 13, 14, 15]], 4))
    env.close()


def drive(env, s, target, tol=0.012, cap=80):
    """Kinematically move robot to a target, returning resulting observation."""
    for _ in range(cap):
        d = np.asarray(target) - s[16:18]
        if np.linalg.norm(d) < tol:
            break
        a = np.clip(d, -0.049, 0.049).astype(np.float32)
        s, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    return s


def local(v, th):
    c, q = math.cos(th), math.sin(th)
    return np.array([c * v[0] - q * v[1], q * v[0] + c * v[1]])


def experiment(seed, offset, direction, n_push=20, clearance=0.75):
    """Approach from opposite push direction, target an offset from COM, and push."""
    env = make_env()
    s, _ = env.reset(seed=seed)
    initial = s.copy()
    u = local(np.asarray(direction, float), float(s[2]))
    u /= np.linalg.norm(u)
    off = local(np.asarray(offset, float), float(s[2]))
    # Contact distance is deliberately generous; push traverses it.
    staging = s[:2] + off - clearance * u
    s = drive(env, s, staging)
    before = s.copy()
    series = []
    for k in range(n_push):
        s, _, term, trunc, _ = env.step((0.049 * u).astype(np.float32))
        series.append((k + 1, *(s[:3] - initial[:3]), s[5], *s[16:18]))
        if term or trunc:
            break
    print("seed/off/dir", seed, offset, direction)
    print(" initial", np.round(initial[[0, 1, 2, 16, 17]], 4), "stage", np.round(before[[0, 1, 2, 16, 17]], 4))
    print(" k dx dy dtheta omega robotx roboty")
    for row in series:
        print(" ", np.round(row, 5))
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["resets", "probe"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ox", type=float, default=0.0)
    ap.add_argument("--oy", type=float, default=0.0)
    ap.add_argument("--dx", type=float, default=1.0)
    ap.add_argument("--dy", type=float, default=0.0)
    args = ap.parse_args()
    if args.mode == "resets":
        resets(range(20))
    else:
        experiment(args.seed, (args.ox, args.oy), (args.dx, args.dy))
