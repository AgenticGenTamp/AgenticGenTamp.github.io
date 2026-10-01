"""Search reproducibly for robot/rod contact in one live episode."""
import numpy as np
from env_client import make_env


env = make_env()
state, info = env.reset(seed=1)
robot = state.get_object_from_name("robot")
names = sorted(n for n in state.get_object_names() if n.startswith("cuboid_"))


def get(obj, feature):
    return float(state.get(obj, feature))


def point(name):
    obj = state.get_object_from_name(name)
    return np.array([get(obj, f) for f in ("x", "y", "z")])


initial = {n: point(n) for n in names}


def step(action, tag):
    global state
    state, reward, term, trunc, _ = env.step(np.asarray(action, np.float32))
    movement = {n: float(np.linalg.norm(point(n) - initial[n])) for n in names}
    if max(movement.values()) > 0.004:
        q = [get(robot, f"pos_arm_joint{i}") for i in range(1, 8)]
        base = [get(robot, f) for f in ("pos_base_x", "pos_base_y", "pos_base_rot")]
        print("HIT", tag, "base", np.round(base, 5), "q", np.round(q, 5),
              "grip", get(robot, "pos_gripper"), "movement", movement,
              "xyz", {n: np.round(point(n), 5) for n in names}, flush=True)
        env.close()
        raise SystemExit


print("objects", {n: np.round(initial[n], 4) for n in names}, flush=True)
# First test a direct base sweep through each rod's centerline with arm at home.
for name in names:
    target = initial[name]
    for phase, goal in enumerate(((-0.15, target[1]), (target[0] + 0.25, target[1]))):
        for k in range(20):
            err = np.asarray(goal) - np.array([get(robot, "pos_base_x"), get(robot, "pos_base_y")])
            if np.max(np.abs(err)) < 0.015:
                break
            a = np.zeros(11); a[:2] = np.clip(err / 0.87, -0.1, 0.1); a[10] = 1.0
            step(a, ("base", name, phase, k))

print("NO HIT", flush=True)
env.close()
