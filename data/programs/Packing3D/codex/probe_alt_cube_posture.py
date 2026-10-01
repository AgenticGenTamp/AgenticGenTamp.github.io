"""Probe alternate final-cuboid wrist postures on count-3 seed 0.

The stock controller packs both triangles.  This subclass changes only the
post-grasp carry/descent of the remaining cuboid.  Arguments are absolute
offsets from PICK_JOINTS for joints 2/4/6, extra negative base-x standoff,
and an absolute joint-2 seating target.
"""
import argparse
import numpy as np

from env_client import make_env
from approach import GeneratedApproach


def value(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


class PostureProbe(GeneratedApproach):
    q2_lift = -.30
    q4_offset = 0.
    q6_offset = 0.
    standoff = .04
    row = -.07
    seat_q2 = .33
    release_axis = 9
    release_pulse = .05
    release_q7 = False

    def reset(self, state, info):
        super().reset(state, info)
        self.alt_started = False
        self.alt_descents = 0

    def _final_cube(self, state):
        return (self.target is not None and len(self.placed) >= 2 and
                state.get_object_from_name(self.target).type.name ==
                "Kinematic3DCuboid")

    def get_action(self, state):
        # Use the proven controller through grasp and its initial vertical lift.
        if not self._final_cube(state) or self.phase not in (
                "carry", "alt_carry", "alt_descend", "alt_release"):
            action = super().get_action(state)
            if self._final_cube(state) and self.phase == "carry":
                self.phase = "alt_carry"
            return action

        joints = self.PICK_JOINTS.copy()
        joints[1] += self.q2_lift
        joints[3] += self.q4_offset
        joints[5] += self.q6_offset
        # Right (+x) rack column and negative-y row.  standoff is deliberately
        # larger than the production controller's 0.02354 m correction.
        base = np.array([self.rack[0] + .04 - self.OFFSET[0] - self.standoff,
                         self.rack[1] + self.row - self.OFFSET[1]], np.float32)
        holding = value(state, "robot", "grasp_active") > .5
        if self.phase == "alt_carry":
            if not self._at(state, base, joints):
                return self._action(state, base, joints, 0.)
            self.phase = "alt_descend"
        if self.phase == "alt_descend":
            if not holding:
                self.phase = "select"
                return self._action(state, grip=1.)
            current = value(state, "robot", "joint_2")
            if current < self.seat_q2 - .004 and self.alt_descents < 25:
                self.alt_descents += 1
                action = self._action(state, grip=0.)
                action[4] = min(.02, self.seat_q2 - current)
                return action
            self.phase = "alt_release"
        if self.phase == "alt_release":
            if not holding:
                self.phase = "select"
                return self._action(state, grip=1.)
            action = self._action(state, grip=1.)
            action[self.release_axis] = self.release_pulse
            if self.release_q7:
                action[9] = .05
            return action
        return self._action(state, grip=1.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q2-lift", type=float, default=-.30)
    parser.add_argument("--q4", type=float, default=0.)
    parser.add_argument("--q6", type=float, default=0.)
    parser.add_argument("--standoff", type=float, default=.04)
    parser.add_argument("--seat", type=float, default=.33)
    parser.add_argument("--row", type=float, default=-.07)
    parser.add_argument("--release-axis", type=int, default=9)
    parser.add_argument("--release-pulse", type=float, default=.05)
    parser.add_argument("--release-q7", action="store_true")
    parser.add_argument("--limit", type=int, default=130)
    args = parser.parse_args()
    for key, val in (("q2_lift", args.q2_lift), ("q4_offset", args.q4),
                     ("q6_offset", args.q6), ("standoff", args.standoff),
                     ("seat_q2", args.seat), ("row", args.row)):
        setattr(PostureProbe, key, val)
    PostureProbe.release_axis = args.release_axis
    PostureProbe.release_pulse = args.release_pulse
    PostureProbe.release_q7 = args.release_q7
    env = make_env()
    state, info = env.reset(seed=0, options={"object_count": 3})
    policy = PostureProbe(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    term = trunc = False
    for step in range(args.limit):
        state, reward, term, trunc, info = env.step(policy.get_action(state))
        if term or trunc:
            break
    positions = {name: tuple(round(value(state, name, "pose_" + axis), 4)
                             for axis in "xyz")
                 for name in policy._parts(state)}
    joints = tuple(round(value(state, "robot", "joint_" + str(i)), 4)
                   for i in (2, 4, 6))
    print("RESULT", vars(args), "term", term, "steps", step + 1, "phase", policy.phase,
          "placed", policy.placed, "hold", value(state, "robot", "grasp_active"),
          "base", (round(value(state, "robot", "pos_base_x"), 4),
                    round(value(state, "robot", "pos_base_y"), 4)),
          "j246", joints, "pos", positions, flush=True)
    env.close()


if __name__ == "__main__":
    main()
