"""Probe alternate held-cuboid IK on count-3 seed 0.

This diagnostic intentionally leaves approach.py untouched.  It packs both
triangles with the stock policy, intercepts the last cuboid while it is held at
high clearance over the rack's right column, and measures how each arm joint
moves the grasped part.
"""
import argparse
import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def value(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


def part_pose(state, name):
    return np.array([value(state, name, "pose_x"),
                     value(state, name, "pose_y"),
                     value(state, name, "pose_z")])


parser = argparse.ArgumentParser()
parser.add_argument("--drop", action="store_true",
                    help="test a base-compensated straight-down drop")
parser.add_argument("--comp", type=float, default=.002)
parser.add_argument("--release-comp", type=float)
parser.add_argument("--impulse", choices=("j7p", "j7p2", "j7n", "j2p",
                                           "j7j2", "rotp", "rotn", "rotnj4", "rotnj2", "seq_q6p", "seq_q6n", "xneg"),
                    default="j7p")
parser.add_argument("--scale", type=float, default=.05)
args = parser.parse_args()


class RightColumnDrop(GeneratedApproach):
    """Aim high at x=.35 and cancel the j2 descent's negative-x arc."""
    def _choose_slot(self, state, part):
        obj = state.get_object_from_name(part)
        if obj.type.name == "Kinematic3DCuboid" and len(self.placed) >= 2:
            cell = (.035, .07)
            return cell, np.array([self.rack[0] + cell[0],
                                   self.rack[1] + cell[1]], np.float32)
        return super()._choose_slot(state, part)

    def get_action(self, state):
        action = super().get_action(state)
        if (self.target == "part0" and len(self.placed) >= 2 and
                self.phase == "descend" and
                value(state, "robot", "grasp_active") > .5):
            # joint_2's local descent also sweeps the cube toward the packed
            # triangles. Translate the mobile base oppositely to descend nearly
            # vertically in world coordinates.
            action[0] = args.comp
        if (args.release_comp is not None and self.target == "part0" and
                len(self.placed) >= 2 and self.phase == "release" and
                value(state, "robot", "grasp_active") > .5):
            action[:] = 0.
            action[10] = 1.
            if value(state, "part0", "pose_z") > .128:
                action[0] = args.release_comp
                action[6] = -.02
            elif args.impulse == "j7p": action[9] = .05
            elif args.impulse == "j7p2": action[9] = .2
            elif args.impulse == "j7n": action[9] = -.05
            elif args.impulse == "j2p": action[4] = .01
            elif args.impulse == "j7j2": action[9] = .05; action[4] = .01
            elif args.impulse == "rotp": action[2] = args.scale
            elif args.impulse == "rotn": action[2] = -args.scale
            elif args.impulse == "rotnj4": action[2] = -args.scale; action[6] = -.02
            elif args.impulse == "rotnj2": action[2] = -args.scale; action[4] = .005
            elif args.impulse == "seq_q6p":
                if value(state,"robot","joint_6") < policy.PICK_JOINTS[5]+.019: action[8]=.02
                else: action[2]=-args.scale
            elif args.impulse == "seq_q6n":
                stage=getattr(self,"seq_stage",0)
                if stage==0 and value(state,"part0","pose_z")>.1251: action[8]=-.02
                elif stage==0: self.seq_stage=1; action[9]=.05
                elif stage==1: self.seq_stage=2; action[8]=args.scale
                elif value(state,"part0","pose_z")>.1245: action[2]=-.01; action[4]=.001
                elif stage==2: self.seq_stage=3; action[8]=.01
                else: action[2]=-.01
            elif args.impulse == "xneg": action[0] = -.01
        return np.clip(action, self.low, self.high).astype(np.float32)


if args.drop:
    env = make_env()
    state, info = env.reset(seed=0, options={"object_count": 3})
    policy = RightColumnDrop(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    terminated = truncated = False
    saw_final_hold = False
    detached = False
    for step in range(140):
        action = policy.get_action(state)
        state, _, terminated, truncated, _ = env.step(action)
        final_hold = policy.target == "part0" and len(policy.placed) >= 2
        held = value(state, "robot", "grasp_active") > .5
        saw_final_hold |= final_hold and held
        if saw_final_hold and final_hold and not held:
            detached = True
            # Let contact/support bookkeeping settle without immediately
            # regrasping the released cube.
            for _ in range(5):
                settle = np.zeros(11, np.float32); settle[10] = 1.
                state, _, terminated, truncated, _ = env.step(settle)
                if terminated or truncated:
                    break
            break
        if terminated or truncated:
            break
    print("DROP", args.comp, "term", terminated, "steps", step + 1,
          "placed", policy.placed, "phase", policy.phase,
          "cube", tuple(np.round(part_pose(state, "part0"), 5)),
          "hold", value(state, "robot", "grasp_active"), "detached", detached)
    env.close()
    raise SystemExit


env = make_env()
state, info = env.reset(seed=0, options={"object_count": 3})
policy = GeneratedApproach(env.action_space, env.observation_space, {})
policy.reset(state, info)

# Reach the stock final-cube high-clearance carry posture.
for step in range(180):
    action = policy.get_action(state)
    state, _, terminated, truncated, _ = env.step(action)
    final = policy.target == "part0" and len(policy.placed) == 2
    if final and policy.phase == "carry" and value(state, "robot", "grasp_active") > .5:
        break
else:
    raise RuntimeError("did not reach held final cuboid")

# Shift the held cube to the empty/right column at high clearance.
right_base = np.array([-.12, .06], np.float32)
lifted = policy.PICK_JOINTS.copy(); lifted[1] -= .30
for _ in range(20):
    if policy._at(state, right_base, lifted):
        break
    state, _, _, _, _ = env.step(policy._action(state, right_base, lifted, 0.))

print("HIGH", "base", tuple(round(value(state, "robot", f), 5)
                              for f in ("pos_base_x", "pos_base_y")),
      "joints", tuple(round(value(state, "robot", "joint_%d" % i), 5)
                      for i in range(1, 8)),
      "cube", tuple(np.round(part_pose(state, "part0"), 5)))
for p in ("part1", "part2"):
    print(p, tuple(np.round(part_pose(state, p), 5)))

# At this clearance +/-0.08 rad pulses are collision-free.  Reverse each pulse
# before testing the next coordinate, producing local Cartesian derivatives.
for joint in range(7):
    before = part_pose(state, "part0")
    action = np.zeros(11, np.float32); action[3 + joint] = .08
    state, _, _, _, _ = env.step(action)
    after = part_pose(state, "part0")
    print("J%d+" % (joint + 1), tuple(np.round(after - before, 6)),
          "pose", tuple(np.round(after, 5)), "hold", value(state, "robot", "grasp_active"))
    action[3 + joint] = -.08
    state, _, _, _, _ = env.step(action)

env.close()
