"""Regrasp a packed triangle before placing seed-0's final cuboid.

This is an experimental subclass only; it does not modify the submitted policy.
"""
import argparse
import numpy as np

from approach import GeneratedApproach
from env_client import make_env


class RelocateApproach(GeneratedApproach):
    relocate_index = 0
    relocate_cell = (-.075, -.075)
    cube_cell = (.015, -.07)
    regrasp_j2 = 0.
    regrasp_j4 = 0.
    extra_descent = 0

    def reset(self, state, info):
        super().reset(state, info)
        self.relocation_started = False
        self.relocation_done = False
        self.relocated_name = None
        self.extra_descent_count = 0

    def _choose_slot(self, state, part):
        obj = state.get_object_from_name(part)
        if obj.type.name == "Kinematic3DCuboid" and self.placed:
            cell = self.cube_cell
            return cell, self.rack + np.asarray(cell, np.float32)
        return super()._choose_slot(state, part)

    def get_action(self, state):
        if self.rack is None:
            self.rack = np.array([self._g(state, "rack", "pose_x"),
                                  self._g(state, "rack", "pose_y")], np.float32)
        if self.total_parts is None:
            self.total_parts = len(self._parts(state))

        # Once two triangles are resting on the rack, explicitly remove one from
        # bookkeeping and feed it through the normal feedback grasp controller.
        if (self.phase == "select" and not self.relocation_started and
                len(self.placed) >= 2):
            triangles = [p for p in self.placed
                         if state.get_object_from_name(p).type.name ==
                         "Kinematic3DTriangle"]
            if len(triangles) >= 2:
                index = min(self.relocate_index, len(triangles) - 1)
                self.relocated_name = triangles[index]
                old_index = self.placed.index(self.relocated_name)
                self.placed.pop(old_index)
                if old_index < len(self.used_slots):
                    self.used_slots.pop(old_index)
                self.target = self.relocated_name
                self.retry = 0
                self.slot_key = self.relocate_cell
                self.slot = self.rack + np.asarray(self.relocate_cell,
                                                   np.float32)
                self.lift_amount = .30
                self.phase = "approach"
                self.relocation_started = True

        # Mark the successful second placement so it cannot trigger again.
        if (self.relocation_started and not self.relocation_done and
                self.relocated_name in self.placed):
            self.relocation_done = True

        # A rack-top grasp attaches roughly 3 cm lower on the mesh than the
        # original table grasp.  Probe additional post-trajectory descent before
        # issuing the release pulse.
        holding = self._g(state, "robot", "grasp_active") > .5
        if (self.phase == "release" and holding and self.relocation_started and
                not self.relocation_done and
                self.target == self.relocated_name and
                self.extra_descent_count < self.extra_descent):
            self.extra_descent_count += 1
            action = self._action(state, grip=0.)
            action[4] = .02
            return action

        # Probe small joint calibration offsets only during the rack regrasp.
        old = self.PICK_JOINTS
        if (self.relocation_started and not self.relocation_done and
                self.target == self.relocated_name and self.phase in
                ("approach", "close")):
            adjusted = old.copy()
            adjusted[1] += self.regrasp_j2
            adjusted[3] += self.regrasp_j4
            self.PICK_JOINTS = adjusted
        action = super().get_action(state)
        self.PICK_JOINTS = old
        return action


def details(state, name):
    obj = state.get_object_from_name(name)
    kind = "cube" if obj.type.name == "Kinematic3DCuboid" else \
        "t%d" % round(state.get(obj, "triangle_type"))
    return (name, kind) + tuple(round(state.get(obj, f), 4) for f in
                                ("pose_x", "pose_y", "pose_z",
                                 "grasp_active"))


def run(limit):
    env = make_env()
    state, info = env.reset(seed=0, options={"object_count": 3})
    policy = RelocateApproach(env.action_space, env.observation_space,
                              env.make_primitives())
    policy.reset(state, info)
    names = sorted(n for n in state.get_object_names() if n.startswith("part"))
    last = (policy.phase, policy.target)
    events = []
    terminated = truncated = False
    for step in range(limit):
        action = policy.get_action(state)
        now = (policy.phase, policy.target)
        if now != last and policy.relocation_started:
            events.append((step, now, details(state, policy.target),
                           tuple(round(float(x), 3) for x in action)))
        last = now
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    print("term", terminated, "steps", step + 1,
          "relocated", policy.relocated_name, policy.relocation_done,
          "placed", policy.placed, "failures", policy.failures)
    print("final", [details(state, n) for n in names])
    print("events", events)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type=int, default=0)
    parser.add_argument("--x", type=float, default=-.075)
    parser.add_argument("--y", type=float, default=-.075)
    parser.add_argument("--cube-x", type=float, default=.015)
    parser.add_argument("--cube-y", type=float, default=-.07)
    parser.add_argument("--j2", type=float, default=0.)
    parser.add_argument("--j4", type=float, default=0.)
    parser.add_argument("--extra", type=int, default=0)
    parser.add_argument("--limit", type=int, default=180)
    args = parser.parse_args()
    RelocateApproach.relocate_index = args.which
    RelocateApproach.relocate_cell = (args.x, args.y)
    RelocateApproach.cube_cell = (args.cube_x, args.cube_y)
    RelocateApproach.regrasp_j2 = args.j2
    RelocateApproach.regrasp_j4 = args.j4
    RelocateApproach.extra_descent = args.extra
    run(args.limit)
