"""Probe collision-safe transitions from released blocker to green0."""
import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach


def vals(p, s):
    joints = [p.g(s, "robot", "joint_" + str(i)) for i in range(1, 8)]
    return p.robot(s), np.asarray(joints), p.g(s, "robot", "base_rot")


seed = int(sys.argv[1])
mode = sys.argv[2] if len(sys.argv) > 2 else "stock"
env = make_env()
s, info = env.reset(seed=seed)
p = GeneratedApproach(env.action_space, env.observation_space, {})
p.reset(s, info)
for k in range(100):
    oldstage = p.stage
    if mode in ("liftmove", "line") and p.stage == 5:
        target = p.target(p.green)
        if not hasattr(p, "probe_phase"):
            p.probe_phase = "move"
            if mode == "line":
                p.probe_route = [p.target(p.green + a * p.out) for a in (.31, .11, 0.)]
        move_target = p.probe_route[0] if mode == "line" and p.probe_route else target
        if p.probe_phase == "move" and p.at(s, move_target):
            if mode == "line" and len(p.probe_route) > 1:
                p.probe_route.pop(0)
                move_target = p.probe_route[0]
            else:
                p.probe_phase = "lower"
        if p.probe_phase == "lower" and p.at(s, target, True):
            p.stage = 6
        if p.stage != 5:
            a = p.get_action(s)
        elif p.probe_phase == "move":
            a = p.motion(s, move_target, 1, lift=True)
        else:
            a = p.motion(s, target, 1)
    else:
        a = p.get_action(s)
    before = vals(p, s)
    s2, _, term, trunc, _ = env.step(a)
    after = vals(p, s2)
    if oldstage >= 4:
        moved = np.max(np.abs(np.r_[after[0]-before[0], after[1]-before[1], after[2]-before[2]]))
        print(k, "stage", oldstage, "=>", p.stage, "moved", round(float(moved), 6),
              "base", after[0], "rot", round(after[2], 4), "q", np.round(after[1], 4),
              "act", np.round(a, 4))
    s = s2
    if term or trunc or k > (99 if mode != "stock" else 35):
        break
print("theta", p.theta, "out", p.out, "off", p.off,
      "green", p.green, "green_target", p.target(p.green), "block", p.blocker)
env.close()
