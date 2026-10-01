"""Search explicit cube-first layouts for seed 0 with three parts.

This is intentionally separate from approach.py.  It supports cube grasp
offsets as well as explicit slots for the cube and two type-1 triangles.
"""
import argparse
import numpy as np

from env_client import make_env
from probe_count3_order import OrderedApproach, describe


class CubeFirstProbe(OrderedApproach):
    order = ("cube", "t1")
    cube_grasp = (0.0, 0.0)
    triangle_grasp = (0.0, 0.0)

    def _shape_correction(self, state, part):
        correction = super()._shape_correction(state, part)
        if self._kind(state, part) == "cube":
            correction = correction + np.asarray(self.cube_grasp, np.float32)
        else:
            correction = correction + np.asarray(self.triangle_grasp, np.float32)
        return correction


def run(seed, limit):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": 3})
    policy = CubeFirstProbe(env.action_space, env.observation_space,
                            env.make_primitives())
    policy.reset(state, info)
    names = sorted(n for n in state.get_object_names() if n.startswith("part"))
    events = []
    old = None
    terminated = truncated = False
    for step in range(limit):
        action = policy.get_action(state)
        marker = (policy.target, policy.phase, tuple(policy.placed))
        state, _, terminated, truncated, _ = env.step(action)
        if marker != old and (marker[1] in ("approach", "lift", "release") or
                              marker[2] != (old[2] if old else ())):
            events.append((step + 1, marker, tuple(round(float(x), 3)
                                                   for x in action)))
        old = marker
        if terminated or truncated:
            break
    print("term", terminated, "steps", step + 1, "grasp", policy.cube_grasp,
          "tri_grasp", policy.triangle_grasp,
          "slots", policy.slots, "events", events,
          "final", [describe(state, n) for n in names],
          "placed", policy.placed, "failures", policy.failures, flush=True)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp", default="0,0")
    parser.add_argument("--tri-grasp", default="0,0")
    parser.add_argument("--slots", default=".04,-.07;-.03,.05;-.03,-.07")
    parser.add_argument("--limit", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    CubeFirstProbe.cube_grasp = tuple(float(v) for v in args.grasp.split(","))
    CubeFirstProbe.triangle_grasp = tuple(float(v) for v in
                                          args.tri_grasp.split(","))
    CubeFirstProbe.slots = tuple(tuple(float(v) for v in c.split(","))
                                 for c in args.slots.split(";"))
    run(args.seed, args.limit)
