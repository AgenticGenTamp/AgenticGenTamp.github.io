"""Raster the base under a fixed low arm pose and repeatedly attempt grasp."""
import numpy as np
from env_client import make_env


env = make_env()
state, _ = env.reset(seed=1)
rod = state.get_object_from_name("cuboid_1")
robot = state.get_object_from_name("robot")


def val(obj, feature):
    return float(state.get(obj, feature))


def rod_xyz():
    return np.array([val(rod, f) for f in ("x", "y", "z")])


def do(action):
    global state
    state, reward, term, trunc, info = env.step(np.asarray(action, np.float32))
    return reward, term, trunc


initial = rod_xyz()
target_q = np.array([0.0, 0.329, 0.0, 0.329, 0.0, 0.0, np.pi / 2])
for _ in range(130):
    q = np.array([val(robot, f"pos_arm_joint{i}") for i in range(1, 8)])
    action = np.zeros(11)
    action[3:10] = np.clip(1.2 * (target_q - q), -0.1, 0.1)
    action[10] = 1.0  # visually confirmed open
    do(action)
    if np.max(np.abs(q - target_q)) < 0.02:
        break
print("READY q", np.round(q, 4), "target", target_q, flush=True)

xs = np.linspace(-0.20, 0.80, 13)
ys = np.linspace(-0.20, 0.45, 10)
samples = 0
for row, y in enumerate(ys):
    row_xs = xs if row % 2 == 0 else xs[::-1]
    for x in row_xs:
        # Closed-loop move to the grid point with gripper open. Base action 0.1
        # corresponds to about 0.087 m per step.
        for _ in range(4):
            err = np.array([x - val(robot, "pos_base_x"), y - val(robot, "pos_base_y")])
            if np.max(np.abs(err)) < 0.018:
                break
            action = np.zeros(11)
            action[:2] = np.clip(err / 0.87, -0.1, 0.1)
            action[10] = 1.0
            do(action)
        # Explicit close, then a tiny x jiggle to expose attachment.
        action = np.zeros(11); action[10] = 0.0; do(action)
        action = np.zeros(11); action[0] = 0.03; action[10] = 0.0; do(action)
        samples += 1
        moved = float(np.linalg.norm(rod_xyz() - initial))
        if moved > 0.003:
            print("HIT", samples, "base", val(robot, "pos_base_x"),
                  val(robot, "pos_base_y"), "rod", rod_xyz(), "move", moved, flush=True)
            env.close()
            raise SystemExit
        # Re-open before moving to the next point.
        action = np.zeros(11); action[10] = 1.0; do(action)
    print("row", row, "y", y, "samples", samples, flush=True)

print("NO HIT", samples, "final rod", rod_xyz(), flush=True)
env.close()
