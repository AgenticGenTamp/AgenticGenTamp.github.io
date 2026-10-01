"""Probe the reproducible third-cuboid release on seed 0 / three parts.

Usage: python probe_crowded_cube_release.py MODE
Modes alter only the final cuboid's descent/release after the stock controller
has successfully packed the first two parts.
"""
import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach


MODE = sys.argv[1] if len(sys.argv) > 1 else "trace"


def get(s, name, feat):
    return float(s.get(s.get_object_from_name(name), feat))


class Probe(GeneratedApproach):
    def reset(self, state, info):
        super().reset(state, info)
        self.extra_descent = 0

    def get_action(self, s):
        pre_final = (self.target is not None and len(self.placed) >= 2 and
                     s.get_object_from_name(self.target).type.name == "Kinematic3DCuboid")
        if pre_final:
            self.slot = np.array([self.rack[0], self.rack[1] - .07], np.float32)
            self.slot_key = (0., -.07)
        # Let the tested policy do all selection, grasping, and transport.
        a = super().get_action(s)
        final_cube = (self.target is not None and len(self.placed) >= 2 and
                      s.get_object_from_name(self.target).type.name == "Kinematic3DCuboid")
        if not final_cube:
            return a
        # Pin the historical failing pose regardless of concurrent experiments in
        # approach.py: this yields object pose about (0.330, -0.070, 0.124).
        self.slot = np.array([self.rack[0], self.rack[1] - .07], np.float32)
        self.slot_key = (0., -.07)
        # Override the stock final-cube contact strategy.  The phase seen here is
        # after super() has advanced its counters, hence descents is 1-based.
        if self.phase == "descend" and MODE == "small":
                if self.descents <= 30:
                    a[:] = 0.; a[4] = .01
                else:
                    self.phase = "release"; a[:] = 0.
        if self.phase == "release":
            if MODE.startswith("n") and self.extra_descent < int(MODE[1:]) - 18:
                # Continue pressing with the gripper unchanged before opening.
                self.extra_descent += 1
                a[:] = 0.; a[4] = .02
                return np.clip(a, self.low, self.high).astype(np.float32)
            a[:] = 0.; a[10] = 1.
            if MODE in ("j2", "small", "j7j2"): a[4] = .01
            elif MODE == "j2neg": a[4] = -.01
            elif MODE == "j4neg": a[6] = -.02
            elif MODE == "j4pos": a[6] = .02
            elif MODE == "j7neg": a[9] = -.05
            if MODE in ("j7pos", "j7j2", "j7xneg", "j7yneg", "j7ypos"):
                a[9] = .05
            if MODE in ("basexneg", "j7xneg"): a[0] = -.01
            if MODE in ("baseyneg", "j7yneg"): a[1] = -.01
            if MODE in ("baseypos", "j7ypos"): a[1] = .01
            # nXX modes open without the stock j7 pulse.
        return np.clip(a, self.low, self.high).astype(np.float32)


env = make_env()
s, info = env.reset(seed=0, options={"object_count": 3})
p = Probe(env.action_space, env.observation_space, {})
p.reset(s, info)
old = None
final_seen_holding = False
for k in range(190):
    action = p.get_action(s)
    phase, target = p.phase, p.target
    s, reward, term, trunc, info = env.step(action)
    if target == "part0" and len(p.placed) >= 2 and get(s, "robot", "grasp_active") > .5:
        final_seen_holding = True
    if target is not None:
        pos = tuple(round(get(s, target, q), 4) for q in ("pose_x", "pose_y", "pose_z"))
    else:
        pos = None
    marker = (phase, target, get(s, "robot", "grasp_active") > .5)
    if marker != old or (target == "part0" and phase in ("descend", "release")) or term:
        print(MODE, k + 1, phase, target, "placed", p.placed,
              "hold", int(get(s, "robot", "grasp_active") > .5),
              "j2", round(get(s, "robot", "joint_2"), 4), "pos", pos,
              "a", tuple(round(float(x), 3) for x in action), "term", term, flush=True)
    old = marker
    if term or trunc or (final_seen_holding and target == "part0" and
                         get(s, "robot", "grasp_active") < .5):
        print("SUMMARY", MODE, "step", k + 1, "phase", phase,
              "hold", int(get(s, "robot", "grasp_active") > .5),
              "pos", pos, "supported", p._supported(s, target), "term", term)
        break
env.close()
