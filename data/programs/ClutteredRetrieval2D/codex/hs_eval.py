"""Evaluate experimental reactive policies without importing approach.py."""

import argparse
import math
import numpy as np
from env_client import make_env


def v(s, o, f):
    return float(s.get(o, f))


def one(s, typ):
    # env_client requires the Type object, while the evaluation state API may
    # also accept strings. Name scanning works in both places.
    return next(s.get_object_from_name(n) for n in s.get_object_names()
                if getattr(s.get_object_from_name(n).type, "name", "") == typ)


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


class Policy:
    def __init__(self, mode="base", stand=0.28):
        self.mode = mode
        self.stand = stand

    def reset(self, s):
        b = one(s, "target_block")
        r = one(s, "crv_robot")
        self.last_b = np.array([v(s, b, "x"), v(s, b, "y")])
        self.last_r = np.array([v(s, r, "x"), v(s, r, "y"),
                                v(s, r, "theta"), v(s, r, "arm_joint")])
        self.carry = False
        self.stuck = 0
        self.step = 0
        self.first_move = None

    def act(self, s):
        self.step += 1
        r, b, g = one(s, "crv_robot"), one(s, "target_block"), one(s, "target_region")
        bp = np.array([v(s, b, "x"), v(s, b, "y")])
        rstate = np.array([v(s, r, "x"), v(s, r, "y"),
                           v(s, r, "theta"), v(s, r, "arm_joint")])
        self.stuck = self.stuck + 1 if np.linalg.norm(rstate - self.last_r) < 1e-6 else 0
        self.last_r = rstate
        if np.linalg.norm(bp - self.last_b) > 1e-5:
            if self.first_move is None:
                self.first_move = self.step
            # Grasped objects tend to move rigidly; for now any motion switches phase.
            self.carry = True
        self.last_b = bp.copy()
        if self.carry:
            gp = np.array([v(s, g, "x"), v(s, g, "y")])
            delta = gp - bp
            a = np.zeros(5, dtype=np.float32)
            angle_error = wrap(v(s, g, "theta") - v(s, b, "theta"))
            if abs(angle_error) > 0.02:
                a[2] = np.clip(angle_error, -0.19634954, 0.19634954)
            else:
                a[0] = np.clip(delta[0], -0.05, 0.05)
                a[1] = np.clip(delta[1], -0.05, 0.05)
            # A grasped block only satisfies the goal once released.
            a[4] = 0.0 if np.linalg.norm(delta) < 0.04 and abs(angle_error) < 0.02 else 1.0
            return a
        target = g if self.carry else b
        tp = np.array([v(s, target, "x"), v(s, target, "y")])
        rp = np.array([v(s, r, "x"), v(s, r, "y")])
        th = v(s, r, "theta")
        vec = tp - rp
        dist = np.linalg.norm(vec)
        desired = math.atan2(vec[1], vec[0]) + 0.08
        err = wrap(desired - th)
        a = np.zeros(5, dtype=np.float32)
        a[4] = 1.0
        # Keep the end effector retracted while driving. Stop base at stand-off.
        joint = v(s, r, "arm_joint")
        if self.stuck >= 2:
            # Contact may already have created a grasp. Pull back to reveal it;
            # if not attached this also resets the approach for another try.
            if joint > 0.105:
                a[3] = -0.05
            else:
                unit = vec / max(dist, 1e-9)
                a[0], a[1] = -0.05 * unit
            self.stuck = 0
            return a
        if self.mode in ("base", "simul"):
            a[3] = np.clip(0.1 - joint, -0.1, 0.1)
        # Translation direction is holonomic and independent of heading.
        move = max(0.0, dist - self.stand)
        if move > 0:
            a[0] = np.clip(vec[0], -0.05, 0.05)
            a[1] = np.clip(vec[1], -0.05, 0.05)
        a[2] = np.clip(err, -0.19634954, 0.19634954)
        # Once aligned and base is close, extend along ray until contact.
        if abs(err) < 0.035 and dist <= self.stand + 0.02:
            # Hypothesis: gripper center radius ~= base_radius + joint + half gripper height.
            want_joint = np.clip(dist - v(s, r, "base_radius") - 0.035, 0.1, 1.0)
            a[3] = np.clip(want_joint - joint, -0.1, 0.1)
        return a


def run(seed, count, mode, stand, cap=1000):
    env = make_env()
    opts = None if count < 0 else {"object_count": count}
    s, _ = env.reset(seed=seed, options=opts)
    p = Policy(mode, stand)
    p.reset(s)
    term = trunc = False
    for t in range(cap):
        s, rew, term, trunc, info = env.step(p.act(s))
        if term or trunc:
            break
    b, g = one(s, "target_block"), one(s, "target_region")
    bd = math.hypot(v(s, b, "x") - v(s, g, "x"), v(s, b, "y") - v(s, g, "y"))
    env.close()
    return term, t + 1, p.first_move, round(bd, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--count", type=int, default=-1)
    ap.add_argument("--mode", default="base")
    ap.add_argument("--stand", type=float, default=0.28)
    args = ap.parse_args()
    results = []
    for seed in range(args.seeds):
        x = run(seed, args.count, args.mode, args.stand)
        results.append(x)
        print(seed, x)
    print("success", sum(x[0] for x in results), "/", len(results),
          "mean solved steps", np.mean([x[1] for x in results if x[0]]) if any(x[0] for x in results) else None)


if __name__ == "__main__":
    main()
