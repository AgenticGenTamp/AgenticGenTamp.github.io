"""Probe explicit three-part placement orders and rack slot sequences.

This deliberately subclasses the submitted approach so experiments do not alter
approach.py.  Example:
  python probe_count3_order.py --order cube,t1,t0 \
      --slots '0.04,-0.07;-0.04,0.05;0.06,0.05' 5
"""
import argparse
import numpy as np

from approach import GeneratedApproach
from env_client import make_env


class OrderedApproach(GeneratedApproach):
    order = ("cube", "t1", "t0")
    slots = ((.04, -.07), (-.03, .05), (.07, .05))

    def _kind(self, state, part):
        obj = state.get_object_from_name(part)
        if obj.type.name == "Kinematic3DCuboid":
            return "cube"
        return "t%d" % round(self._g(state, part, "triangle_type"))

    def _choose_slot(self, state, part):
        index = min(len(self.placed), len(self.slots) - 1)
        cell = self.slots[index]
        return cell, np.array([self.rack[0] + cell[0],
                               self.rack[1] + cell[1]], np.float32)

    def get_action(self, state):
        # Install an explicit target before delegating the normal controller.
        if self.rack is None:
            self.rack = np.array([self._g(state, "rack", "pose_x"),
                                  self._g(state, "rack", "pose_y")], np.float32)
        if self.total_parts is None:
            self.total_parts = len(self._parts(state))
        if self.phase == "select":
            candidates = [p for p in self._parts(state)
                          if p not in self.placed and not self._supported(state, p)]
            if candidates:
                priority = {kind: i for i, kind in enumerate(self.order)}
                self.target = min(candidates,
                                  key=lambda p: (priority.get(self._kind(state, p), 99), p))
                self.retry = 0
                self.phase = "approach"
                self.slot_key, self.slot = self._choose_slot(state, self.target)
                self.lift_amount = .30 if self.placed else .15
        return super().get_action(state)


def describe(state, name):
    obj = state.get_object_from_name(name)
    kind = "cube" if obj.type.name == "Kinematic3DCuboid" else \
        "t%d" % round(state.get(obj, "triangle_type"))
    return (name, kind, round(state.get(obj, "pose_x"), 3),
            round(state.get(obj, "pose_y"), 3),
            round(state.get(obj, "pose_z"), 3),
            round(state.get(obj, "grasp_active")))


def run(seed, limit):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": 3})
    policy = OrderedApproach(env.action_space, env.observation_space,
                             env.make_primitives())
    policy.reset(state, info)
    names = sorted(n for n in state.get_object_names() if n.startswith("part"))
    initial = [describe(state, name) for name in names]
    chosen = []
    old_target = None
    terminated = truncated = False
    for step in range(limit):
        action = policy.get_action(state)
        if policy.target != old_target:
            chosen.append((policy.target, policy._kind(state, policy.target),
                           policy.slot_key))
            old_target = policy.target
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    print("seed", seed, "term", terminated, "steps", step + 1,
          "initial", initial, "chosen", chosen,
          "final", [describe(state, name) for name in names],
          "placed", policy.placed, "failures", policy.failures)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", default="cube,t1,t0")
    parser.add_argument("--slots", default=".04,-.07;-.03,.05;.07,.05")
    parser.add_argument("--limit", type=int, default=180)
    parser.add_argument("seeds", nargs="*", type=int, default=list(range(10)))
    args = parser.parse_args()
    OrderedApproach.order = tuple(args.order.split(","))
    OrderedApproach.slots = tuple(tuple(float(v) for v in cell.split(","))
                                  for cell in args.slots.split(";"))
    for episode_seed in args.seeds:
        run(episode_seed, args.limit)
