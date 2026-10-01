"""Staged sweep-grasp policy experiment for ClutteredRetrieval2D."""

import argparse
import math
import numpy as np
from env_client import make_env
from hs_eval import one, v, wrap


class StagedPolicy:
    def reset(self, s):
        r, b = one(s, "crv_robot"), one(s, "target_block")
        rp = np.array([v(s, r, "x"), v(s, r, "y")])
        bp = np.array([v(s, b, "x"), v(s, b, "y")])
        self.phi = math.atan2(bp[1] - rp[1], bp[0] - rp[0])
        self.u = np.array([math.cos(self.phi), math.sin(self.phi)])
        self.pre = bp - 0.30 * self.u
        self.last_b = bp
        self.phase = "pre"

    def act(self, s):
        r, b, g = one(s, "crv_robot"), one(s, "target_block"), one(s, "target_region")
        rp = np.array([v(s, r, "x"), v(s, r, "y")])
        bp = np.array([v(s, b, "x"), v(s, b, "y")])
        gp = np.array([v(s, g, "x"), v(s, g, "y")])
        th = v(s, r, "theta")
        moved = np.linalg.norm(bp - self.last_b) > 1e-5
        self.last_b = bp.copy()
        if moved and self.phase in ("pre", "sweep"):
            self.phase = "retreat"
            self.retreat = bp - 0.40 * self.u

        a = np.zeros(5, dtype=np.float32)
        a[4] = 1.0
        a[3] = np.clip(0.1 - v(s, r, "arm_joint"), -0.1, 0.1)
        if self.phase == "pre":
            d = self.pre - rp
            a[0:2] = np.clip(d, -0.05, 0.05)
            a[2] = np.clip(wrap(self.phi - 0.70 - th), -0.19634954, 0.19634954)
            if np.linalg.norm(d) < 0.01 and abs(wrap(self.phi - 0.70 - th)) < 0.02:
                self.phase = "sweep"
        elif self.phase == "sweep":
            # Sweep fully across the target; depending on target orientation,
            # first contact can occur just before or just after centerline.
            a[2] = np.clip(wrap(self.phi + 0.70 - th), -0.10, 0.10)
        elif self.phase == "retreat":
            d = self.retreat - bp
            a[0:2] = np.clip(d, -0.05, 0.05)
            a[3] = 0.0
            if np.linalg.norm(d) < 0.02:
                self.phase = "align"
        elif self.phase == "align":
            err = wrap(v(s, g, "theta") - v(s, b, "theta"))
            a[2] = np.clip(err, -0.10, 0.10)
            a[3] = 0.0
            if abs(err) < 0.015:
                self.phase = "deliver"
        elif self.phase == "deliver":
            d = gp - bp
            a[0:2] = np.clip(d, -0.05, 0.05)
            a[3] = 0.0
            if np.linalg.norm(d) < 0.015:
                a[0:2] = 0.0
                a[4] = 0.0
                self.phase = "release"
        elif self.phase == "release":
            a[4] = 0.0
            a[3] = 0.0
        return a


def run(seed, count, cap=1000):
    env = make_env()
    s, _ = env.reset(seed=seed, options={"object_count": count})
    p = StagedPolicy(); p.reset(s)
    te = tr = False
    for t in range(cap):
        s, _, te, tr, _ = env.step(p.act(s))
        if te or tr:
            break
    b, g = one(s, "target_block"), one(s, "target_region")
    bd = math.hypot(v(s, b, "x") - v(s, g, "x"), v(s, b, "y") - v(s, g, "y"))
    out = te, t + 1, p.phase, round(bd, 3)
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--count", type=int, default=0); args = ap.parse_args()
    xs = []
    for seed in range(args.seeds):
        x = run(seed, args.count); xs.append(x); print(seed, x)
    print("success", sum(x[0] for x in xs), "/", len(xs), "mean", np.mean([x[1] for x in xs if x[0]]) if any(x[0] for x in xs) else None)


if __name__ == "__main__":
    main()
