"""Survey Motion2DEnv initial-state geometry over a range of seeds."""

import math
import sys
from collections import Counter

import numpy as np

from env_client import make_env


def value(state, obj, feature):
    return float(state.get(obj, feature))


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    env = make_env()
    rows = []
    for seed in range(count):
        state, info = env.reset(seed=seed)
        robots = state.get_objects(env.observation_space.get_type("crv_robot"))
        targets = state.get_objects(env.observation_space.get_type("target_region"))
        rects = state.get_objects(env.observation_space.get_type("rectangle"))
        robot = robots[0]
        target = targets[0]
        # target_region is a rectangle subtype too; identify count-defining objects
        # by their documented names rather than treating all returned rectangles as
        # obstacles.
        obstacles = [o for o in rects if str(o.name).startswith("obstacle")]
        rx, ry = value(state, robot, "x"), value(state, robot, "y")
        tx, ty = value(state, target, "x"), value(state, target, "y")
        direct = math.hypot(tx - rx, ty - ry)
        rows.append(
            {
                "seed": seed,
                "n": len(obstacles),
                "rx": rx,
                "ry": ry,
                "rtheta": value(state, robot, "theta"),
                "radius": value(state, robot, "base_radius"),
                "arm_joint": value(state, robot, "arm_joint"),
                "arm_length": value(state, robot, "arm_length"),
                "gripper_height": value(state, robot, "gripper_height"),
                "gripper_width": value(state, robot, "gripper_width"),
                "tx": tx,
                "ty": ty,
                "ttheta": value(state, target, "theta"),
                "tw": value(state, target, "width"),
                "th": value(state, target, "height"),
                "direct": direct,
                "obs": [
                    (
                        value(state, o, "x"), value(state, o, "y"),
                        value(state, o, "theta"), value(state, o, "width"),
                        value(state, o, "height"),
                    )
                    for o in obstacles
                ],
            }
        )
    env.close()

    print("count", count)
    print("obstacle_count", sorted(Counter(r["n"] for r in rows).items()))
    for k in ("rx", "ry", "rtheta", "radius", "arm_joint", "arm_length",
              "gripper_height", "gripper_width", "tx", "ty", "ttheta",
              "tw", "th", "direct"):
        a = np.asarray([r[k] for r in rows])
        print(k, "min/median/max", *(f"{x:.6g}" for x in (a.min(), np.median(a), a.max())),
              "unique", len(np.unique(np.round(a, 8))))

    all_obs = [o for r in rows for o in r["obs"]]
    for i, k in enumerate(("ox", "oy", "otheta", "ow", "oh")):
        a = np.asarray([o[i] for o in all_obs])
        print(k, "min/median/max", *(f"{x:.6g}" for x in (a.min(), np.median(a), a.max())),
              "unique", len(np.unique(np.round(a, 8))))

    gaps = []
    layout_metrics = []
    for r in rows:
        by_x = {}
        for o in r["obs"]:
            by_x.setdefault(round(o[0], 6), []).append(o)
        for x, pieces in by_x.items():
            if len(pieces) == 2:
                pieces.sort(key=lambda o: o[1])
                raw_gap = pieces[1][1] - (pieces[0][1] + pieces[0][4])
                gaps.append((raw_gap, r["seed"], x, pieces[0][1] + pieces[0][4], pieces[1][1]))
        ordered = []
        direct_ok = True
        min_feasible = 99.0
        for x, pieces in sorted(by_x.items()):
            if len(pieces) != 2:
                continue
            pieces.sort(key=lambda o: o[1])
            lo = pieces[0][1] + pieces[0][4] + r["radius"]
            hi = pieces[1][1] - r["radius"]
            clipped_lo = max(r["radius"], lo)
            clipped_hi = min(2.5 - r["radius"], hi)
            min_feasible = min(min_feasible, clipped_hi - clipped_lo)
            ordered.append((x, (clipped_lo + clipped_hi) / 2.0))
            direct_y = r["ry"] + (r["ty"] - r["ry"]) * (x - r["rx"]) / (r["tx"] - r["rx"])
            direct_ok = direct_ok and clipped_lo <= direct_y <= clipped_hi
        ys = [r["ry"]] + [p[1] for p in ordered] + [r["ty"]]
        zigzag = sum(abs(a - b) for a, b in zip(ys, ys[1:]))
        layout_metrics.append((zigzag, min_feasible if ordered else 99.0, direct_ok, r["seed"], r["n"], ys))
    if gaps:
        a = np.asarray([g[0] for g in gaps])
        print("wall_gap min/median/max", *(f"{x:.6g}" for x in (a.min(), np.median(a), a.max())))
        print("narrowest_gaps", [tuple(round(v, 5) if isinstance(v, float) else v for v in g) for g in sorted(gaps)[:20]])
        print("direct_base_path_fraction", sum(m[2] for m in layout_metrics) / len(layout_metrics))
        print("tightest_clipped", [(round(m[1], 5), m[3], m[4], [round(y, 3) for y in m[5]])
                                   for m in sorted(layout_metrics, key=lambda m: m[1])[:20]])
        print("most_zigzag", [(round(m[0], 4), m[3], m[4], round(m[1], 4), [round(y, 3) for y in m[5]])
                              for m in sorted(layout_metrics, reverse=True)[:20]])

    # Rank approximate difficulty indicators independent of exact collision logic.
    print("farthest", [(r["seed"], round(r["direct"], 3), r["n"]) for r in sorted(rows, key=lambda x: -x["direct"])[:15]])
    print("most_obstacles", [(r["seed"], r["n"], round(r["direct"], 3)) for r in sorted(rows, key=lambda x: (-x["n"], -x["direct"]))[:20]])
    # Compact exact dumps for a few seeds with the most obstacles.
    for r in sorted(rows, key=lambda x: (-x["n"], -x["direct"]))[:8]:
        print("SEED", r["seed"], "robot", tuple(round(r[k], 4) for k in ("rx", "ry", "radius")),
              "target", tuple(round(r[k], 4) for k in ("tx", "ty", "tw", "th", "ttheta")),
              "obs", [tuple(round(v, 4) for v in o) for o in r["obs"]])


if __name__ == "__main__":
    main()
