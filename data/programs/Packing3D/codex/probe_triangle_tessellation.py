"""Test complementary rotated type-1 triangles on count-3 seed 0.

The submitted policy is left untouched.  This subclass places both triangles at
one rack location, rotating the second about the grasp axis before transport,
then puts the cuboid in the other row.
"""
import argparse
import numpy as np

from approach import GeneratedApproach
from env_client import make_env


class TessellationApproach(GeneratedApproach):
    angle = np.pi
    first_cell = (0.0, 0.055)
    second_delta = (0.0, 0.0)
    cube_cell = (0.0, -0.07)

    def reset(self, state, info):
        super().reset(state, info)
        self.rotated_target = None
        self.rotation_done = False

    def _kind(self, state, part):
        obj = state.get_object_from_name(part)
        return "cube" if obj.type.name == "Kinematic3DCuboid" else "triangle"

    def _choose_slot(self, state, part):
        kind = self._kind(state, part)
        if kind == "cube":
            cell = self.cube_cell
        elif not any(self._kind(state, p) == "triangle" for p in self.placed):
            cell = self.first_cell
        else:
            cell = (self.first_cell[0] + self.second_delta[0],
                    self.first_cell[1] + self.second_delta[1])
            self.rotated_target = part
        return cell, self.rack + np.asarray(cell, np.float32)

    def get_action(self, state):
        # Explicit order: triangle, triangle, cube.
        if self.rack is None:
            self.rack = np.array([self._g(state, "rack", "pose_x"),
                                  self._g(state, "rack", "pose_y")], np.float32)
        if self.total_parts is None:
            self.total_parts = len(self._parts(state))
        if self.phase == "select":
            candidates = [p for p in self._parts(state)
                          if p not in self.placed and not self._supported(state, p)]
            if candidates:
                self.target = min(candidates,
                                  key=lambda p: (self._kind(state, p) == "cube", p))
                self.retry = 0
                self.phase = "approach"
                self.slot_key, self.slot = self._choose_slot(state, self.target)
                self.lift_amount = .30 if self.placed else .15

        holding = self._g(state, "robot", "grasp_active") > .5
        if self.target == self.rotated_target and holding and self.phase == "lift":
            lifted = self.PICK_JOINTS.copy()
            lifted[1] -= self.lift_amount
            lifted[6] += self.angle
            if not self._at(state, joints=lifted):
                return self._action(state, joints=lifted, grip=0.)
            self.phase = "carry"
            self.rotation_done = True

        if self.target == self.rotated_target and holding and self.phase == "carry":
            lifted = self.PICK_JOINTS.copy()
            lifted[1] -= self.lift_amount
            lifted[6] += self.angle
            part_xy = np.array([self._g(state, self.target, "pose_x"),
                                self._g(state, self.target, "pose_y")], np.float32)
            error = self.slot - part_xy
            base = np.array([self._g(state, "robot", "pos_base_x"),
                             self._g(state, "robot", "pos_base_y")], np.float32)
            if np.max(np.abs(error)) > .003:
                return self._action(state, base + error, lifted, 0.)
            self.phase = "descend"
            self.descents = 0
        return super().get_action(state)


def pose(state, name):
    obj = state.get_object_from_name(name)
    fields = ("pose_x", "pose_y", "pose_z", "pose_qx", "pose_qy",
              "pose_qz", "pose_qw", "grasp_active")
    return tuple(round(float(state.get(obj, f)), 5) for f in fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=float, default=float(np.pi))
    parser.add_argument("--x", type=float, default=0.)
    parser.add_argument("--y", type=float, default=.055)
    parser.add_argument("--dx", type=float, default=0.)
    parser.add_argument("--dy", type=float, default=0.)
    parser.add_argument("--cube-x", type=float, default=0.)
    parser.add_argument("--cube-y", type=float, default=-.07)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=180)
    args = parser.parse_args()
    TessellationApproach.angle = args.angle
    TessellationApproach.first_cell = (args.x, args.y)
    TessellationApproach.second_delta = (args.dx, args.dy)
    TessellationApproach.cube_cell = (args.cube_x, args.cube_y)

    env = make_env()
    state, info = env.reset(seed=args.seed, options={"object_count": 3})
    policy = TessellationApproach(env.action_space, env.observation_space,
                                  env.make_primitives())
    policy.reset(state, info)
    old_phase = old_target = None
    events = []
    terminated = truncated = False
    for step in range(args.limit):
        action = policy.get_action(state)
        if policy.phase != old_phase or policy.target != old_target:
            events.append((step, policy.target, policy.phase,
                           tuple(round(float(x), 4) for x in action)))
            old_phase, old_target = policy.phase, policy.target
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    print("seed", args.seed, "term", terminated, "trunc", truncated, "steps", step + 1,
          "placed", policy.placed, "failures", policy.failures)
    print("poses", [(p, pose(state, p)) for p in policy._parts(state)])
    print("events", events)
    env.close()


if __name__ == "__main__":
    main()
