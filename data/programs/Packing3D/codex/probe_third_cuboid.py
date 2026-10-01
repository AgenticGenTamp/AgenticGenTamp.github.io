"""Probe the third-part cuboid's rack target without modifying approach.py."""
import argparse
import numpy as np

from approach import GeneratedApproach
from env_client import make_env


class CuboidTargetApproach(GeneratedApproach):
    cuboid_slot = (0.04, -0.07)
    carry_bias = (0.0, 0.0)
    triangle_x = None
    post_release_push = 0

    def reset(self, state, info):
        super().reset(state, info)
        self.push_count = 0

    def get_action(self, state):
        holding = self._g(state, "robot", "grasp_active") > .5
        if (self.phase == "release" and not holding and self.target and
                state.get_object_from_name(self.target).type.name ==
                "Kinematic3DCuboid" and
                self.push_count < self.post_release_push):
            self.push_count += 1
            action = self._action(state, grip=1.)
            action[0] = -.02
            return action
        return super().get_action(state)

    def _choose_slot(self, state, part):
        obj = state.get_object_from_name(part)
        if (obj.type.name == "Kinematic3DTriangle" and
                self.triangle_x is not None):
            rows = (-.07, .05)
            row = rows[min(len(self.placed), 1)]
            cell = (self.triangle_x, row)
            return cell, np.array([self.rack[0] + cell[0],
                                   self.rack[1] + cell[1]], np.float32)
        if obj.type.name == "Kinematic3DCuboid" and self.placed:
            cell = self.cuboid_slot
            target = np.array([self.rack[0] + cell[0] + self.carry_bias[0],
                               self.rack[1] + cell[1] + self.carry_bias[1]],
                              np.float32)
            return cell, target
        return super()._choose_slot(state, part)


def pose(state, name):
    obj = state.get_object_from_name(name)
    return tuple(round(state.get(obj, key), 5)
                 for key in ("pose_x", "pose_y", "pose_z", "grasp_active"))


def run(seed, limit):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": 3})
    policy = CuboidTargetApproach(env.action_space, env.observation_space,
                                  env.make_primitives())
    policy.reset(state, info)
    old_phase = policy.phase
    events = []
    terminated = truncated = False
    for step in range(limit):
        action = policy.get_action(state)
        if policy.target is not None:
            obj = state.get_object_from_name(policy.target)
            if obj.type.name == "Kinematic3DCuboid" and policy.phase != old_phase:
                robot = state.get_object_from_name("robot")
                events.append((step, policy.phase, pose(state, policy.target),
                               round(state.get(robot, "pos_base_x"), 5),
                               round(state.get(robot, "pos_base_y"), 5),
                               tuple(round(float(x), 3) for x in action)))
        old_phase = policy.phase
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    print("seed", seed, "slot", CuboidTargetApproach.cuboid_slot,
          "bias", CuboidTargetApproach.carry_bias, "term", terminated,
          "steps", step + 1, "final", pose(state, "part0"),
          "placed", policy.placed, "failures", policy.failures)
    print("events", events)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=.04)
    parser.add_argument("--y", type=float, default=-.07)
    parser.add_argument("--bias-x", type=float, default=0.)
    parser.add_argument("--bias-y", type=float, default=0.)
    parser.add_argument("--triangle-x", type=float)
    parser.add_argument("--push", type=int, default=0)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("seeds", type=int, nargs="*", default=[0])
    args = parser.parse_args()
    CuboidTargetApproach.cuboid_slot = (args.x, args.y)
    CuboidTargetApproach.carry_bias = (args.bias_x, args.bias_y)
    CuboidTargetApproach.triangle_x = args.triangle_x
    CuboidTargetApproach.post_release_push = args.push
    for requested_seed in args.seeds:
        run(requested_seed, args.limit)
