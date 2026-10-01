"""Sweep one/two-joint alternate postures while preserving the modeled tool pose."""
import itertools
import sys

import numpy as np

from ik_shallow_probe import setup, g
from probe_grasp import fk


def run(seed, changes):
    e, s, p = setup(seed)
    base0 = np.r_[p.robot(s), g(s, "robot", "base_rot")]
    goalq = p.Q.copy()
    for j, d in changes:
        goalq[j] += d
    # Preserve nominal-Q modeled tool xy by translating the base.
    nominal = fk(base0, p.Q)[:3, 3]
    alt = fk(base0, goalq)[:3, 3]
    goalbase = base0[:2] + nominal[:2] - alt[:2]
    if np.any(np.abs(goalbase) > 5):
        e.close(); return None
    # First establish redundancy with arm lifted, then descend all joints and
    # compensate base. Close throughout so a near crossing is captured.
    liftq = goalq.copy(); liftq[1] -= .2
    for goal_b, q in ((goalbase, liftq), (goalbase, goalq)):
        for _ in range(25):
            a = np.zeros(11, np.float32)
            here = p.robot(s); a[:2] = np.clip(goal_b-here, -.05, .05)
            for j in range(7):
                d = q[j]-g(s, "robot", "joint_"+str(j+1))
                if j in (4, 6): d = p.w(d)
                a[3+j] = np.clip(d, -.04, .04)
            a[10] = -1
            old = np.r_[here, [g(s, "robot", "joint_"+str(j+1)) for j in range(7)]]
            s, *_ = e.step(a)
            if g(s, "green0", "grasp_active") > .5:
                vals = [g(s, "robot", "joint_"+str(j+1)) for j in range(7)]
                e.close(); return goalbase, goalq, vals
            new = np.r_[p.robot(s), [g(s, "robot", "joint_"+str(j+1)) for j in range(7)]]
            if max(abs(new-old)) < 1e-7: break
    e.close(); return False


seed = int(sys.argv[1]) if len(sys.argv) > 1 else 101
vals = [-.24, -.16, -.08, .08, .16, .24]
# shoulder pan, upper roll, elbow flex, forearm roll, wrist flex
for j in (0, 2, 3, 4, 5):
    for d in vals:
        z = run(seed, [(j, d)])
        if z:
            print("HIT", seed, [(j, d)], z, flush=True); raise SystemExit
        print("try", j, d, z, flush=True)
for j1, j2 in ((0, 3), (2, 3), (3, 4), (3, 5)):
    for d1, d2 in itertools.product((-.16, -.08, .08, .16), repeat=2):
        z = run(seed, [(j1, d1), (j2, d2)])
        if z:
            print("HIT", seed, [(j1, d1), (j2, d2)], z, flush=True); raise SystemExit
print("NONE")
