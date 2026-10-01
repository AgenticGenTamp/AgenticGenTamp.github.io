"""Sweep deterministic grasp approaches for a difficult single-block seed."""

import math
import sys

import numpy as np

from env_client import make_env


def angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def run(seed, offset, vacuum_during_extension=True, stand_off=0.54):
    env = make_env()
    state, _ = env.reset(seed=seed)
    robot = state.get_object_from_name("robot")
    blocks = list(state.get_objects(env.observation_space.get_type("target_block")))
    get = lambda o, f: float(state.get(o, f))
    shelf = state.get_object_from_name("shelf")
    sx, sy = get(shelf, "x1"), get(shelf, "y1")
    sw, sh = get(shelf, "width1"), get(shelf, "height1")
    outside = [b for b in blocks if not
               (sx <= get(b, "x") <= sx+sw and sy <= get(b, "y") <= sy+sh)]
    pool = outside or blocks
    block = min(pool, key=lambda b: (get(b, "x")-get(robot, "x"))**2 +
                                    (get(b, "y")-get(robot, "y"))**2)
    initial = (get(block, "x"), get(block, "y"), get(block, "theta"))
    target_theta = angle(initial[2] + offset)
    goal = (initial[0] - stand_off * math.cos(target_theta),
            initial[1] - stand_off * math.sin(target_theta))

    # Simultaneously retract, align, and translate to the computed base pose.
    for _ in range(100):
        rx, ry, rt, arm = (get(robot, f) for f in
                           ("x", "y", "theta", "arm_joint"))
        ex, ey, et = goal[0] - rx, goal[1] - ry, angle(target_theta - rt)
        if max(abs(ex), abs(ey)) < .006 and abs(et) < .008 and abs(arm-.2) < .006:
            break
        action = np.array([np.clip(ex, -.05, .05), np.clip(ey, -.05, .05),
                           np.clip(et, -.196, .196), np.clip(.2-arm, -.1, .1), 0],
                          dtype=np.float32)
        state, _, _, _, _ = env.step(action)

    acquired_at = None
    trace = []
    retracting = False
    contact_position = None
    followed_retraction = False
    for tick in range(14):
        bx, by, bt = (get(block, f) for f in ("x", "y", "theta"))
        arm = get(robot, "arm_joint")
        vac = 1.0 if vacuum_during_extension else float(tick >= 6)
        da = -.1 if retracting else (.1 if tick < 6 else 0)
        action = np.array([0, 0, 0, da, vac], dtype=np.float32)
        state, _, _, _, _ = env.step(action)
        now = tuple(get(block, f) for f in ("x", "y", "theta"))
        moved = math.hypot(now[0]-initial[0], now[1]-initial[1]) > .003
        moved |= abs(angle(now[2]-initial[2])) > .01
        trace.append((tick, round(arm, 3), tuple(round(v, 4) for v in now), moved))
        if moved and acquired_at is None:
            acquired_at = tick
            contact_position = now[:2]
            retracting = True
        elif retracting and contact_position is not None:
            followed_retraction |= math.hypot(now[0]-contact_position[0],
                                              now[1]-contact_position[1]) > .03
    final_robot = tuple(round(get(robot, f), 4) for f in
                        ("x", "y", "theta", "arm_joint", "vacuum"))
    env.close()
    return acquired_at, followed_retraction, initial, goal, final_robot, trace


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 27
    # Relative to the long block axis: both end-on and both broad-side normals.
    offsets = [0, math.pi, math.pi/2, -math.pi/2,
               math.pi/4, -math.pi/4, 3*math.pi/4, -3*math.pi/4]
    for off in offsets:
        result = run(seed, off)
        print("offset", round(off, 4), "contact", result[0],
              "attached", result[1],
              "block", tuple(round(v, 4) for v in result[2]),
              "goal", tuple(round(v, 4) for v in result[3]),
              "robot", result[4])
        if result[0] is not None:
            print(" trace", result[5])
