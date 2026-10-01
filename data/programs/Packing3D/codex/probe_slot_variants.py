"""Compare candidate rack coordinates without changing the submitted policy."""
import argparse
import numpy as np

from approach import GeneratedApproach
from env_client import make_env


class WideRowsApproach(GeneratedApproach):
    """Move triangle rows farther apart from previously placed cuboids."""
    def _choose_slot(self, state, part):
        obj = state.get_object_from_name(part)
        if obj.type.name == "Kinematic3DCuboid":
            cells = [(self.cuboid_x, -.07), (self.cuboid_x, .07)]
        else:
            cells = [(self.triangle_x, self.negative_row),
                     (self.second_triangle_x, self.positive_row)]
        free = [cell for cell in cells
                if not any(abs(cell[0] - used[0]) < .07 and
                           abs(cell[1] - used[1]) < .06
                           for used in self.used_slots)] or cells
        cell = free[0]
        return cell, np.array([self.rack[0] + cell[0],
                               self.rack[1] + cell[1]], np.float32)

    def _supported(self, state, part):
        if self.triangles_first:
            obj = state.get_object_from_name(part)
            if obj.type.name == "Kinematic3DCuboid":
                for name in self._parts(state):
                    other = state.get_object_from_name(name)
                    if (other.type.name == "Kinematic3DTriangle" and
                            state.get(other, "triangle_type") > .5 and
                            name not in self.placed and
                            not super()._supported(state, name)):
                        return True
        return super()._supported(state, part)


def run(seed, count, limit=180):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": count})
    policy = WideRowsApproach(env.action_space, env.observation_space,
                              env.make_primitives())
    policy.reset(state, info)
    terminated = truncated = False
    for step in range(limit):
        state, _, terminated, truncated, _ = env.step(policy.get_action(state))
        if terminated or truncated:
            break
    parts = []
    for name in sorted(n for n in state.get_object_names() if n.startswith("part")):
        obj = state.get_object_from_name(name)
        kind = obj.type.name
        if kind == "Kinematic3DTriangle":
            kind = "tri%d" % round(state.get(obj, "triangle_type"))
        parts.append((kind, round(state.get(obj, "pose_x"), 3),
                      round(state.get(obj, "pose_y"), 3),
                      round(state.get(obj, "pose_z"), 3)))
    print(seed, count, terminated, step + 1, parts, policy.used_slots,
          policy.failures)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--positive-row", type=float, default=.11)
    parser.add_argument("--negative-row", type=float, default=-.09)
    parser.add_argument("--triangle-x", type=float, default=-.03)
    parser.add_argument("--second-triangle-x", type=float, default=-.03)
    parser.add_argument("--cuboid-x", type=float, default=0.)
    parser.add_argument("--triangles-first", action="store_true")
    parser.add_argument("seeds", nargs="*", type=int, default=[1, 2])
    args = parser.parse_args()
    WideRowsApproach.positive_row = args.positive_row
    WideRowsApproach.negative_row = args.negative_row
    WideRowsApproach.triangle_x = args.triangle_x
    WideRowsApproach.second_triangle_x = args.second_triangle_x
    WideRowsApproach.cuboid_x = args.cuboid_x
    WideRowsApproach.triangles_first = args.triangles_first
    for seed in args.seeds:
        run(seed, args.count)
