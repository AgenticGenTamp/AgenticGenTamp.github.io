"""Randomized local search for any successful grasp near the default pose."""
from env_client import make_env
import numpy as np


def g(s, o, f): return float(s.get(o, f))


def main(seed=0, trials=120):
    rng = np.random.default_rng(260915)
    env = make_env()
    s, _ = env.reset(seed=seed)
    rob = s.get_object_from_name("robot")
    part = s.get_object_from_name("part0")
    rack = s.get_object_from_name("rack")
    default = np.array([g(s, rob, f"joint_{i}") for i in range(1, 8)])
    align_x = g(s, rob, "pos_base_x") + g(s, part, "pose_x") - g(s, rack, "pose_x")
    align_y = g(s, part, "pose_y")
    for trial in range(trials):
        # Search shoulder/elbow/wrist pitch near the downward default pose.
        target = default.copy()
        target[[1, 3, 5]] += rng.uniform(-0.75, 0.75, 3)
        bx = align_x + rng.uniform(-0.14, 0.02)  # +x is tightly bounded
        by = align_y + rng.uniform(-0.14, 0.14)
        for _ in range(5):
            a = np.zeros(11, dtype=np.float32)
            a[0] = np.clip(bx-g(s, rob, "pos_base_x"), -.2, .2)
            a[1] = np.clip(by-g(s, rob, "pos_base_y"), -.2, .2)
            for k in range(7):
                a[3+k] = np.clip(target[k]-g(s, rob, f"joint_{k+1}"), -.2, .2)
            a[10] = 1
            s, *_ = env.step(a)
        a = np.zeros(11, dtype=np.float32); a[10] = -1
        s, *_ = env.step(a)
        if g(s, rob, "grasp_active"):
            js = [g(s, rob, f"joint_{i}") for i in range(1, 8)]
            base = [g(s, rob, f"pos_base_{q}") for q in ("x", "y", "rot")]
            tf = [g(s, rob, f"grasp_tf_{q}") for q in ("x", "y", "z")]
            print("SUCCESS", trial, "base", base, "joints", js, "tf", tf)
            env.close(); return
        if trial % 10 == 0: print("trial", trial, flush=True)
    print("none")
    env.close()


if __name__ == "__main__": main()
