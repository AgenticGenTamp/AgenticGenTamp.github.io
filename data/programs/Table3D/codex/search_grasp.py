"""Empirically search joint configurations that trigger a grasp."""

import argparse
import numpy as np
from env_client import make_env


def val(s, obj, feature):
    return s.get(obj, feature)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=140)
    args = ap.parse_args()
    rng = np.random.default_rng(12345)
    env = make_env()
    s, _ = env.reset(seed=args.seed)
    robot = s.get_object_from_name("robot")
    # Broad Panda-like limits. Keep wrist angles less extreme than full limits.
    lo = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.1, -2.8])
    hi = np.array([ 2.8,  1.7,  2.8, -0.1,  2.8,  3.6,  2.8])
    for trial in range(args.trials):
        target = rng.uniform(lo, hi)
        # Move toward each random configuration using bounded deltas.
        for _ in range(8):
            cur = np.array([val(s, robot, f"joint_{i}") for i in range(1, 8)])
            a = np.zeros(11, dtype=np.float32)
            a[3:10] = np.clip(target - cur, -0.4, 0.4)
            a[10] = 1.0
            s, _, term, trunc, _ = env.step(a)
            robot = s.get_object_from_name("robot")
            if term or trunc: break
        a = np.zeros(11, dtype=np.float32); a[10] = -1.0
        s, _, term, trunc, _ = env.step(a)
        robot = s.get_object_from_name("robot")
        if val(s, robot, "grasp_active") > 0.5:
            q = [val(s, robot, f"joint_{i}") for i in range(1, 8)]
            print("GRASP", trial, q)
            for n in sorted(s.get_object_names()):
                if n.startswith("cube"):
                    o=s.get_object_from_name(n)
                    if val(s,o,"grasp_active")>.5: print("OBJECT", n)
            break
        if term or trunc: break
    else:
        print("NO_GRASP")
    print("steps", trial * 9 + 9, "done", term, trunc)
    env.close()


if __name__ == "__main__": main()
