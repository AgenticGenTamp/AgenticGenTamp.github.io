"""Sweep selected arm joints through a floor rod and report any contact."""
import sys

import numpy as np
from env_client import make_env


joint = int(sys.argv[1]) if len(sys.argv) > 1 else 1
closed = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
amplitude = float(sys.argv[4]) if len(sys.argv) > 4 else 0.30
base_dx = float(sys.argv[5]) if len(sys.argv) > 5 else 0.
base_dy = float(sys.argv[6]) if len(sys.argv) > 6 else 0.
cycles = int(sys.argv[7]) if len(sys.argv) > 7 else 3

env = make_env()
state, _ = env.reset(seed=seed)
robot = state.get_object_from_name("robot")
movable = env.observation_space.get_type("mujoco_movable_object")
rods = sorted(state.get_objects(movable), key=lambda o: o.name)
rod = rods[min(1, len(rods) - 1)]


def val(obj, feature):
    return float(state.get(obj, feature))


def xyz(obj):
    return np.array([val(obj, f) for f in ("x", "y", "z")])


def advance(action):
    global state
    state, reward, term, trunc, _ = env.step(np.asarray(action, np.float32))
    return reward, term or trunc


initial = {o.name: xyz(o) for o in rods}
# Exact joint state from the visually lowest rendered policy frame.  Joint 3
# is continuous; use its equivalent wrapped angle to avoid an unnecessary turn.
pick = np.array([-.28471, .62423, 1.27544, -.83415,
                 .35352, -.04872, -1.33449])
rod0 = initial[rod.name]
base_target = np.array([rod0[0] - .68 + base_dx,
                        rod0[1] - .11 + base_dy])

# Align and reach the known near-floor pose with fingers open.
for step in range(180):
    q = np.array([val(robot, f"pos_arm_joint{i}") for i in range(1, 8)])
    err_q = pick - q
    err_q[[0, 2, 4, 6]] = (err_q[[0, 2, 4, 6]] + np.pi) % (2*np.pi) - np.pi
    err_b = base_target - np.array([val(robot, "pos_base_x"), val(robot, "pos_base_y")])
    action = np.zeros(11)
    action[:2] = np.clip(err_b / .87, -.1, .1)
    action[3:10] = np.clip(.7 * err_q, -.1, .1)
    action[10] = 1.
    advance(action)
    if max(np.max(abs(err_q)), np.max(abs(err_b))) < .012:
        break

print("READY", "joint", joint, "closed", closed, "base",
      np.round([val(robot, "pos_base_x"), val(robot, "pos_base_y")], 4),
      "q", np.round(q, 4), "rod", np.round(rod0, 4), flush=True)

# Close first when requested. Then repeatedly sweep from -amp to +amp and
# back, using velocity-limited joint control while all other joints hold pose.
for _ in range(4):
    action = np.zeros(11); action[10] = 0. if closed else 1.
    advance(action)

hit = False
for cycle in range(cycles):
    targets = np.r_[np.linspace(-amplitude, amplitude, 21),
                    np.linspace(amplitude, -amplitude, 21)[1:]]
    for sample, offset in enumerate(targets):
        target = pick.copy(); target[joint - 1] += offset
        for _ in range(3):
            q = np.array([val(robot, f"pos_arm_joint{i}") for i in range(1, 8)])
            err = target - q
            err[[0, 2, 4, 6]] = (err[[0, 2, 4, 6]] + np.pi) % (2*np.pi) - np.pi
            action = np.zeros(11)
            action[3:10] = np.clip(.9 * err, -.1, .1)
            action[10] = 0. if closed else 1.
            advance(action)
        moves = {o.name: float(np.linalg.norm(xyz(o) - initial[o.name])) for o in rods}
        if max(moves.values()) > .003:
            print("HIT", "cycle", cycle, "sample", sample, "offset", offset,
                  "q", np.round(q, 5), "moves", moves,
                  "positions", {o.name: np.round(xyz(o), 5).tolist() for o in rods},
                  flush=True)
            hit = True
            break
    if hit:
        break

if not hit:
    moves = {o.name: float(np.linalg.norm(xyz(o) - initial[o.name])) for o in rods}
    print("MISS", "joint", joint, "closed", closed, "moves", moves, flush=True)
env.close()
