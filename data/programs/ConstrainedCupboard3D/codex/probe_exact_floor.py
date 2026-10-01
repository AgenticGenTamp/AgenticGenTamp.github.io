"""Long-settle verification of the corrected rendered floor posture."""
import numpy as np
from env_client import make_env

env = make_env()
state, _ = env.reset(seed=1)
robot = state.get_object_from_name("robot")
rod = state.get_object_from_name("cuboid_1")


def v(obj, feature):
    return float(state.get(obj, feature))


def step(action):
    global state
    state, reward, terminated, truncated, _ = env.step(
        np.asarray(action, np.float32))


def rodpos():
    return np.array([v(rod, f) for f in ("x", "y", "z")])


origin = rodpos()
target = np.array([-.28471, .62423, 1.275445, -.83415,
                   .35352, -.04872, -1.33449])
for _ in range(140):
    q = np.array([v(robot, f"pos_arm_joint{i}") for i in range(1, 8)])
    err = target - q
    err[[0, 2, 4, 6]] = (err[[0, 2, 4, 6]] + np.pi) % (2*np.pi) - np.pi
    action = np.zeros(11)
    action[3:10] = np.clip(.5 * err, -.05, .05)
    action[10] = 1.
    step(action)

goal = origin[:2] - [.68, .11]
for _ in range(20):
    base = np.array([v(robot, "pos_base_x"), v(robot, "pos_base_y")])
    action = np.zeros(11)
    action[:2] = np.clip((goal-base)/.87, -.04, .04)
    action[10] = 1.
    step(action)
for _ in range(20):
    action = np.zeros(11); action[10] = 1.; step(action)
print("OPEN", v(robot, "pos_gripper"), "q", np.round(q, 6),
      "base", np.round(base, 6), "rod", np.round(rodpos(), 6), flush=True)
for k in range(40):
    step(np.zeros(11))
    if k in (0, 4, 9, 19, 39):
        print("CLOSE", k+1, v(robot, "pos_gripper"),
              np.round(rodpos()-origin, 7), flush=True)
for k in range(10):
    action = np.zeros(11); action[0] = .02; step(action)
    print("MOVE", k, np.round(rodpos()-origin, 7), flush=True)
env.close()
