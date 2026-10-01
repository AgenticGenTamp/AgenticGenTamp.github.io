"""Fine wrist/base search around the empirically low arm posture.

Usage: probe_wrist_contact.py dq5 dq6 dq7 [seed]
Each invocation owns a fresh environment so candidates can run in parallel.
"""
import sys
import numpy as np
from env_client import make_env


dq = np.array([float(x) for x in sys.argv[1:4]])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
env = make_env()
state, _ = env.reset(seed=seed)
robot = state.get_object_from_name("robot")
rods = sorted((n for n in state.get_object_names() if n.startswith("cuboid_")))
# Use the middle rod when possible: it is isolated from walls and other rods.
rod = state.get_object_from_name(rods[len(rods) // 2])


def value(obj, feature):
    return float(state.get(obj, feature))


def xyz():
    return np.array([value(rod, f) for f in ("x", "y", "z")])


def advance(action):
    global state
    state, reward, terminated, truncated, _ = env.step(
        np.asarray(action, dtype=np.float32))
    return reward, terminated or truncated


origin = xyz()
max_move = 0.
closest = None
target = np.array([-.28471, .62423, 1.275445, -.83415,
                   .35352, -.04872, -1.33449])
target[4:] += dq

# Settle exactly at the candidate posture, open and away from the objects.
for _ in range(100):
    q = np.array([value(robot, f"pos_arm_joint{i}") for i in range(1, 8)])
    error = target - q
    error[[0, 2, 4, 6]] = (error[[0, 2, 4, 6]] + np.pi) % (2*np.pi) - np.pi
    action = np.zeros(11)
    action[3:10] = np.clip(.8 * error, -.1, .1)
    action[10] = 1.
    advance(action)
    if np.max(np.abs(error)) < .015:
        break

# The nominal wrist offset inferred by the prior visual calibration is
# (+.68,+.11) from the base. Search at 1 cm pitch, finer than finger width.
center = origin[:2] - np.array([.68, .11])
offsets = np.arange(-.04, .041, .01)
for iy, oy in enumerate(offsets):
    row = offsets if iy % 2 == 0 else offsets[::-1]
    for ox in row:
        goal = center + [ox, oy]
        for _ in range(3):
            base = np.array([value(robot, "pos_base_x"),
                             value(robot, "pos_base_y")])
            error = goal - base
            if np.max(np.abs(error)) < .004:
                break
            action = np.zeros(11)
            action[:2] = np.clip(error / .87, -.1, .1)
            action[10] = 1.
            advance(action)

        # Let the fingers close fully, then jiggle in four directions. A true
        # grasp follows the base; incidental resting motion is also reported.
        action = np.zeros(11)
        for _ in range(3):
            advance(action)
        for axis, sign in ((0, 1), (1, 1)):
            action = np.zeros(11)
            action[axis] = .018 * sign
            advance(action)
            move = float(np.linalg.norm(xyz() - origin))
            if move > max_move:
                max_move = move
                closest = (float(ox), float(oy), xyz().copy())
            if move > .003:
                base = [value(robot, f) for f in
                        ("pos_base_x", "pos_base_y", "pos_base_rot")]
                actual_q = [value(robot, f"pos_arm_joint{i}") for i in range(1, 8)]
                print("HIT", "dq", dq, "offset", [ox, oy],
                      "base", np.round(base, 6), "q", np.round(actual_q, 6),
                      "grip", value(robot, "pos_gripper"),
                      "rod0", np.round(origin, 6), "rod", np.round(xyz(), 6),
                      "delta", np.round(xyz() - origin, 6), flush=True)
                env.close()
                raise SystemExit
        action = np.zeros(11)
        action[10] = 1.
        advance(action)

print("MISS", "dq", dq, "q", np.round(q, 5), "origin", np.round(origin, 5),
      "maxmove", max_move, "at", closest,
      "grip", value(robot, "pos_gripper"), flush=True)
env.close()
