"""Evaluate a simple wall-gap waypoint policy without importing approach.py."""
from env_client import make_env
import numpy as np
import math


def val(s, o, f):
    return float(s.get(o, f))


def run(seed, verbose=False):
    env = make_env()
    s, _ = env.reset(seed=seed)
    rt = env.observation_space.get_type("crv_robot")
    tt = env.observation_space.get_type("target_region")
    rect = env.observation_space.get_type("rectangle")
    robot = s.get_objects(rt)[0]
    target = s.get_objects(tt)[0]
    rad = val(s, robot, "base_radius")
    obs = [o for o in s.get_objects(rect) if o.name.startswith("obstacle")]
    groups = {}
    for o in obs:
        groups.setdefault(round(val(s, o, "x"), 5), []).append(o)
    waypoints = []
    for x, parts in sorted(groups.items()):
        if x <= val(s, robot, "x") or x >= val(s, target, "x") + val(s, target, "width"):
            continue
        parts.sort(key=lambda o: min(val(s, o, "y"), val(s, o, "y") + val(s, o, "height")))
        # The generated layout has exactly two segments, with (x,y) at lower left.
        low, high = parts[0], parts[-1]
        lo = max(val(s, low, "y"), val(s, low, "y") + val(s, low, "height")) + rad + .005
        hi = min(val(s, high, "y"), val(s, high, "y") + val(s, high, "height")) - rad - .005
        mid = (lo + hi) / 2
        # Approach the opening while still clear of the wall, then cross it.
        halfw = val(s, low, "width") / 2
        # Extra 0.02 accounts for the simulator's collision skin / endpoint contact.
        waypoints.append((x - halfw - rad - .025, mid))
        waypoints.append((x + halfw + rad + .025, mid))
    # Rectangle x/y is the lower-left for the target; aim at its center.
    waypoints.append((val(s, target, "x") + val(s, target, "width") / 2,
                      val(s, target, "y") + val(s, target, "height") / 2))
    wi = 0
    blocked = 0
    for step in range(env.max_steps):
        robot = s.get_objects(rt)[0]
        x, y = val(s, robot, "x"), val(s, robot, "y")
        gx, gy = waypoints[wi]
        dx, dy = gx - x, gy - y
        if max(abs(dx), abs(dy)) < .01 and wi + 1 < len(waypoints):
            wi += 1
            continue
        theta = val(s, robot, "theta")
        turn = (0.0 - theta + math.pi) % (2 * math.pi) - math.pi
        a = np.asarray([np.clip(dx, -.05, .05), np.clip(dy, -.05, .05),
                        np.clip(turn, -.196, .196), -.1, 0], np.float32)
        ns, rew, term, trunc, _ = env.step(a)
        nr = ns.get_objects(rt)[0]
        if abs(val(ns, nr, "x") - x) + abs(val(ns, nr, "y") - y) < 1e-8:
            blocked += 1
            if verbose and blocked <= 3:
                print("blocked", seed, "at", (x, y), "toward", (gx, gy), "waypoint", wi, "all", waypoints)
                env.close()
                return False, step + 1, blocked, len(obs)
        s = ns
        if term or trunc:
            env.close()
            return term, step + 1, blocked, len(obs)
    env.close()
    return False, env.max_steps, blocked, len(obs)


if __name__ == "__main__":
    results = [run(i) for i in range(40)]
    fails = [(i, x) for i, x in enumerate(results) if not x[0]]
    print("success", len(results) - len(fails), "/", len(results), "mean steps", sum(x[1] for x in results) / len(results))
    print("fails", fails[:20])
