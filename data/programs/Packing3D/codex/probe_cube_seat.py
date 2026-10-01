"""Probe a short open-gripper seating pulse after the final cube detaches."""
import argparse
import numpy as np

from approach import GeneratedApproach
from env_client import make_env


class SeatProbe(GeneratedApproach):
    axis = 4
    pulse = .02
    pulses = 8

    def reset(self, state, info):
        super().reset(state, info)
        self.seating = 0

    def get_action(self, state):
        final_cube = (self.target is not None and len(self.placed) >= 2 and
                      state.get_object_from_name(self.target).type.name ==
                      "Kinematic3DCuboid")
        holding = self._g(state, "robot", "grasp_active") > .5
        if final_cube and not holding and self.phase in ("release", "descend"):
            if self.seating < self.pulses:
                self.seating += 1
                action = self._action(state, grip=1.)
                action[self.axis] = self.pulse
                return np.clip(action, self.low, self.high).astype(np.float32)
        return super().get_action(state)


def run(seed, limit):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": 3})
    policy = SeatProbe(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    term = trunc = False
    for step in range(limit):
        state, _, term, trunc, _ = env.step(policy.get_action(state))
        if term or trunc:
            break
    part = state.get_object_from_name("part0")
    pose = tuple(round(float(state.get(part, "pose_" + q)), 4) for q in "xyz")
    print("seed", seed, "axis", SeatProbe.axis, "pulse", SeatProbe.pulse,
          "pulses", SeatProbe.pulses, "term", term, "steps", step + 1,
          "pose", pose, "held", policy._g(state, "robot", "grasp_active"),
          "placed", policy.placed, flush=True)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", type=int, default=4)
    parser.add_argument("--pulse", type=float, default=.02)
    parser.add_argument("--pulses", type=int, default=8)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("seeds", type=int, nargs="*", default=[2, 4])
    args = parser.parse_args()
    SeatProbe.axis = args.axis
    SeatProbe.pulse = args.pulse
    SeatProbe.pulses = args.pulses
    for seed in args.seeds:
        run(seed, args.limit)
