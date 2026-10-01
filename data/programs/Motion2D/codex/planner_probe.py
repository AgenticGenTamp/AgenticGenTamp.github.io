from env_client import make_env
import math
import numpy as np


def value(s, o, f):
    return float(s.get(o, f))


for seed in range(12):
    env = make_env()
    s, info = env.reset(seed=seed)
    rob = s.get_objects(env.observation_space.get_type("crv_robot"))[0]
    tgt = s.get_objects(env.observation_space.get_type("target_region"))[0]
    rects = s.get_objects(env.observation_space.get_type("rectangle"))
    print("seed", seed, "names", s.get_object_names())
    print(" robot", [round(value(s, rob, f), 4) for f in ("x", "y", "theta", "base_radius", "arm_joint", "arm_length", "gripper_height", "gripper_width")])
    print(" target", [round(value(s, tgt, f), 4) for f in ("x", "y", "theta", "width", "height")])
    print(" rects", [[round(value(s, o, f), 4) for f in ("x", "y", "theta", "width", "height")] for o in rects])
    env.close()


def test_motion(seed, action, n=8):
    env = make_env()
    s, _ = env.reset(seed=seed)
    typ = env.observation_space.get_type("crv_robot")
    rob = s.get_objects(typ)[0]
    pts = [(value(s, rob, "x"), value(s, rob, "y"))]
    for _ in range(n):
        s, r, term, trunc, info = env.step(np.asarray(action, dtype=np.float32))
        rob = s.get_objects(typ)[0]
        pts.append((value(s, rob, "x"), value(s, rob, "y")))
        if term or trunc:
            break
    print("motion", seed, action[:2], [[round(x, 3), round(y, 3)] for x, y in pts], "last", r, term, trunc, info)
    env.close()


for seed in range(4):
    test_motion(seed, [0.05, 0.05, 0, -0.1, 0], 10)


def waypoint_run(seed, waypoints, retract=True):
    env = make_env()
    s, _ = env.reset(seed=seed)
    rtyp = env.observation_space.get_type("crv_robot")
    rob = s.get_objects(rtyp)[0]
    print("waypoint start", seed, "arm_joint", value(s, rob, "arm_joint"), "arm_length", value(s, rob, "arm_length"))
    blocked = 0
    for goal in waypoints:
        for _ in range(100):
            rob = s.get_objects(rtyp)[0]
            x, y = value(s, rob, "x"), value(s, rob, "y")
            dx, dy = goal[0] - x, goal[1] - y
            if max(abs(dx), abs(dy)) < 0.005:
                break
            act = np.asarray([max(-.05, min(.05, dx)), max(-.05, min(.05, dy)), 0, -.1 if retract else 0, 0], np.float32)
            ns, rew, term, trunc, info = env.step(act)
            nr = ns.get_objects(rtyp)[0]
            if abs(value(ns, nr, "x") - x) + abs(value(ns, nr, "y") - y) < 1e-8:
                blocked += 1
            s = ns
            if term or trunc:
                break
    rob = s.get_objects(rtyp)[0]
    print("waypoint end", seed, [round(value(s, rob, q), 4) for q in ("x", "y", "arm_joint", "arm_length")], "blocked", blocked, "term", term)
    env.close()


# Seed 0 first gap has y midpoint 1.65365; compare arm extended/retracted.
waypoint_run(0, [(0.30, 1.65365), (0.75, 1.65365)], True)
waypoint_run(0, [(0.30, 1.65365), (0.75, 1.65365)], False)


# Distribution summary, avoiding assumptions based only on the first few seeds.
counts, thetas, widths, xs = set(), set(), set(), set()
for seed in range(100):
    env = make_env()
    s, _ = env.reset(seed=seed)
    typ = env.observation_space.get_type("rectangle")
    obstacles = [o for o in s.get_objects(typ) if o.name.startswith("obstacle")]
    counts.add(len(obstacles))
    for o in obstacles:
        thetas.add(round(value(s, o, "theta"), 6))
        widths.add(round(value(s, o, "width"), 6))
        xs.add(round(value(s, o, "x"), 6))
    env.close()
print("summary100", "rectangle_counts", sorted(counts), "theta", sorted(thetas), "width", sorted(widths), "x", sorted(xs))
