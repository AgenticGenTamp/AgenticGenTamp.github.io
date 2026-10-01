"""Empirical reset/kinematics survey for ClutteredRetrieval2DEnv."""

import math
import sys
from collections import Counter, defaultdict

import numpy as np

from env_client import make_env


def obj_dict(state, obj):
    return {f: float(state.get(obj, f)) for f in state.type_features[obj.type]}


def main():
    env = make_env()
    type_by_name = {t.name: t for t in env.observation_space.types}
    rows = []
    requested_counts = [None] * 20 + [0, 1, 2, 3, 5, 8, 12]
    for seed, count in enumerate(requested_counts):
        options = None if count is None else {"object_count": count}
        state, info = env.reset(seed=seed, options=options)
        robot = obj_dict(state, state.get_objects(type_by_name["crv_robot"])[0])
        block = obj_dict(state, state.get_objects(type_by_name["target_block"])[0])
        region = obj_dict(state, state.get_objects(type_by_name["target_region"])[0])
        obstacles = [
            obj_dict(state, o)
            for o in state.get_objects(type_by_name["rectangle"])
            if o.type.name == "rectangle"
        ]
        rows.append((seed, count, robot, block, region, obstacles, info))

    def rng(path):
        vals = []
        for row in rows:
            root = row[path[0]]
            vals.append(root[path[1]])
        return min(vals), max(vals), len(set(round(v, 6) for v in vals))

    print("max_steps", env.max_steps)
    print("sampled obstacle counts", Counter(len(r[5]) for r in rows))
    for name, idx in (("robot", 2), ("block", 3), ("region", 4)):
        print(name)
        for feature in rows[0][idx]:
            print(" ", feature, rng((idx, feature)))
    print("obstacles")
    for feature in rows[0][5][0]:
        vals = [o[feature] for r in rows for o in r[5]]
        print(" ", feature, (min(vals), max(vals), len(set(round(v, 6) for v in vals))))

    print("layouts seed requested actual: robot | block | region | obs")
    for seed, count, rob, b, reg, obs, info in rows:
        fmt = lambda d: tuple(round(d[k], 3) for k in ("x", "y", "theta", "width", "height") if k in d)
        print(seed, count, len(obs), fmt(rob), fmt(b), fmt(reg), [fmt(o) for o in obs])

    # Probe unconstrained and possibly colliding action semantics on a clean instance.
    state, _ = env.reset(seed=101, options={"object_count": 0})
    rob_obj = state.get_objects(type_by_name["crv_robot"])[0]
    block_obj = state.get_objects(type_by_name["target_block"])[0]
    print("kinematic probe initial", obj_dict(state, rob_obj), obj_dict(state, block_obj))
    actions = [
        np.array([.05, 0, 0, 0, 0], dtype=np.float32),
        np.array([0, .05, 0, 0, 0], dtype=np.float32),
        np.array([0, 0, math.pi / 16, 0, 0], dtype=np.float32),
        np.array([0, 0, 0, .1, 0], dtype=np.float32),
        np.array([0, 0, 0, -.1, 1], dtype=np.float32),
    ]
    for action in actions:
        state, reward, term, trunc, info = env.step(action)
        print(" after", action.tolist(), obj_dict(state, rob_obj), obj_dict(state, block_obj), reward, term, trunc)
    env.close()


def grasp_probe():
    """Try inferred gripper-center offsets on obstacle-free copies of one seed."""
    env = make_env()
    type_by_name = {t.name: t for t in env.observation_space.types}
    for direction in (-1.0,):
        for desired_arm in (0.0, 0.1, 0.2):
          for distance in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
            state, _ = env.reset(seed=202, options={"object_count": 0})
            robot = state.get_objects(type_by_name["crv_robot"])[0]
            block = state.get_objects(type_by_name["target_block"])[0]
            bx, by = state.get(block, "x"), state.get(block, "y")
            desired_theta = 0.0 if direction > 0 else math.pi
            desired_x = bx - direction * distance
            # Route around the block, because the circular base cannot pass through it.
            desired_y = by + 0.35
            for _ in range(100):
                x, y = state.get(robot, "x"), state.get(robot, "y")
                theta = state.get(robot, "theta")
                err = (desired_theta - theta + math.pi) % (2 * math.pi) - math.pi
                action = np.array([
                    np.clip(desired_x - x, -.05, .05),
                    np.clip(desired_y - y, -.05, .05),
                    np.clip(err, -math.pi / 16, math.pi / 16),
                    np.clip(desired_arm - state.get(robot, "arm_joint"), -.1, .1),
                    0,
                ], dtype=np.float32)
                state, _, _, _, _ = env.step(action)
                if max(abs(desired_x - state.get(robot, "x")), abs(desired_y - state.get(robot, "y")), abs(err)) < 1e-3:
                    break
            # Turn vacuum on before making end-effector contact, then approach sideways.
            for _ in range(10):
                dy = np.clip(by - state.get(robot, "y"), -.05, .05)
                state, _, _, _, _ = env.step(np.array([0, dy, 0, 0, 1], dtype=np.float32))
                if abs(by - state.get(robot, "y")) < 1e-3:
                    break
            before = np.array([state.get(block, "x"), state.get(block, "y")])
            state, _, _, _, _ = env.step(np.array([.03, 0, 0, 0, 1], dtype=np.float32))
            after = np.array([state.get(block, "x"), state.get(block, "y")])
            print("grasp", direction, "joint", desired_arm, "distance", distance,
                  "delta", np.round(after - before, 4),
                  "base", round(state.get(robot, "x"), 3), round(state.get(robot, "y"), 3),
                  "theta", round(state.get(robot, "theta"), 3),
                  "arm", round(state.get(robot, "arm_joint"), 3),
                  "block_pre", np.round(before, 3))
    env.close()


def rectangles_overlap(a, b):
    """Separating-axis test for two dictionaries describing oriented rectangles."""
    def corners(r):
        c, s = math.cos(r["theta"]), math.sin(r["theta"])
        out = []
        for u in (-r["width"] / 2, r["width"] / 2):
            for v in (-r["height"] / 2, r["height"] / 2):
                out.append((r["x"] + c * u - s * v, r["y"] + s * u + c * v))
        return out
    ca, cb = corners(a), corners(b)
    axes = []
    for r in (a, b):
        c, s = math.cos(r["theta"]), math.sin(r["theta"])
        axes.extend(((c, s), (-s, c)))
    for ax, ay in axes:
        pa = [ax * x + ay * y for x, y in ca]
        pb = [ax * x + ay * y for x, y in cb]
        if max(pa) < min(pb) or max(pb) < min(pa):
            return False
    return True


def distribution_probe():
    env = make_env()
    types = {t.name: t for t in env.observation_space.types}
    counts, overlap_counts, obs_dists, rb_dists, br_dists = [], [], [], [], []
    extents = defaultdict(list)
    for seed in range(100):
        state, _ = env.reset(seed=1000 + seed)
        robot = state.get_objects(types["crv_robot"])[0]
        block_obj = state.get_objects(types["target_block"])[0]
        region_obj = state.get_objects(types["target_region"])[0]
        block, region = obj_dict(state, block_obj), obj_dict(state, region_obj)
        obs = [obj_dict(state, o) for o in state.get_objects(types["rectangle"])
               if o.type.name == "rectangle"]
        counts.append(len(obs))
        overlap_counts.append(sum(rectangles_overlap(block, o) for o in obs))
        obs_dists.extend(math.hypot(o["x"] - block["x"], o["y"] - block["y"]) for o in obs)
        rb_dists.append(math.hypot(state.get(robot, "x") - block["x"],
                                  state.get(robot, "y") - block["y"]))
        br_dists.append(math.hypot(region["x"] - block["x"], region["y"] - block["y"]))
        for name, r in (("block", block), ("region", region)):
            # Circumscribed-coordinate extent is a conservative boundary check.
            rad = math.hypot(r["width"], r["height"]) / 2
            extents[name].extend((r["x"] - rad, r["x"] + rad,
                                  r["y"] - rad, r["y"] + rad))
    def q(v):
        return tuple(round(float(x), 3) for x in np.quantile(v, [0, .1, .5, .9, 1]))
    print("counts", Counter(counts))
    print("target-overlapping obstacle count", Counter(overlap_counts), "q", q(overlap_counts))
    print("obstacle center distance to block q", q(obs_dists))
    print("robot-block distance q", q(rb_dists))
    print("block-region distance q", q(br_dists))
    print("conservative coordinate extents", {k: (round(min(v), 3), round(max(v), 3))
                                                for k, v in extents.items()})
    env.close()


def retention_probe():
    """Acquire the clean target, then test translation, rotation, and release."""
    env = make_env()
    types = {t.name: t for t in env.observation_space.types}
    state, _ = env.reset(seed=202, options={"object_count": 0})
    robot = state.get_objects(types["crv_robot"])[0]
    block = state.get_objects(types["target_block"])[0]
    bx, by = state.get(block, "x"), state.get(block, "y")
    desired = (bx + .30, by + .35, math.pi)
    sequence = []
    for _ in range(100):
        rx, ry, th = (state.get(robot, k) for k in ("x", "y", "theta"))
        err = (desired[2] - th + math.pi) % (2 * math.pi) - math.pi
        a = np.array([np.clip(desired[0] - rx, -.05, .05),
                      np.clip(desired[1] - ry, -.05, .05),
                      np.clip(err, -math.pi / 16, math.pi / 16), 0, 0], dtype=np.float32)
        state, _, _, _, _ = env.step(a)
        sequence.append(a)
        if abs(desired[0] - state.get(robot, "x")) < 1e-3 and abs(desired[1] - state.get(robot, "y")) < 1e-3 and abs(err) < 1e-3:
            break
    while state.get(robot, "y") > by + 1e-3:
        a = np.array([0, np.clip(by - state.get(robot, "y"), -.05, .05), 0, 0, 1], dtype=np.float32)
        state, _, _, _, _ = env.step(a)
        sequence.append(a)
    print("acquired? pose", [round(state.get(robot, k), 4) for k in ("x", "y", "theta", "arm_joint")],
          "block", [round(state.get(block, k), 4) for k in ("x", "y", "theta")],
          "route_steps", len(sequence))
    previous = np.array([state.get(block, "x"), state.get(block, "y"), state.get(block, "theta")])
    tests = [(.03, 0, 0, 0, 1), (0, .03, 0, 0, 1), (0, 0, math.pi / 16, 0, 1),
             (0, 0, -math.pi / 16, .1, 1), (.03, 0, 0, 0, 0)]
    for vals in tests:
        state, _, _, _, _ = env.step(np.array(vals, dtype=np.float32))
        now = np.array([state.get(block, "x"), state.get(block, "y"), state.get(block, "theta")])
        print("test", vals, "block_delta", np.round(now - previous, 5),
              "robot", [round(state.get(robot, k), 4) for k in ("x", "y", "theta", "arm_joint", "vacuum")])
        previous = now
    env.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "grasp":
        grasp_probe()
    elif len(sys.argv) > 1 and sys.argv[1] == "dist":
        distribution_probe()
    elif len(sys.argv) > 1 and sys.argv[1] == "retain":
        retention_probe()
    else:
        main()
