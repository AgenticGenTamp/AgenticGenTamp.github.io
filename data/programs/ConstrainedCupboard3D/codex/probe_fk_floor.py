"""Probe floor-reaching Gen3 poses derived from the public Kinova URDF."""

import sys

import numpy as np
from env_client import make_env


POSES = [
    # FK tool position (arm-base frame), assuming 18 cm terminal tool:
    # approximately (0.20, 0.00, -0.20).
    [2.867, 1.088, -0.581, 1.371, -1.412, 2.123, 1.108],
    # approximately (0.30, 0.00, -0.15).
    [-2.835, -1.999, -3.086, 1.829, -0.526, -2.152, 2.195],
    # Rotations of pose A about joint 1, used to find the unobstructed edge of
    # the mobile base's top deck.
    [-0.275, 1.088, -0.581, 1.371, -1.412, 2.123, 1.108],
    [-1.845, 1.088, -0.581, 1.371, -1.412, 2.123, 1.108],
    [1.296, 1.088, -0.581, 1.371, -1.412, 2.123, 1.108],
    # Reoptimized with observed task limits q4 <= -0.314 and q6 <= 1.359.
    # Tool targets are approximately (0.15, 0, -0.10) and
    # (0.20, 0, -0.10), respectively.
    [1.395, 1.044, -2.626, -0.711, 1.519, -1.513, -0.648],
    [0.786, 0.309, -2.858, -0.526, 2.670, -1.611, -0.749],
    # Lowest valid posture found under the observed limits. Tool FK is
    # approximately (0.095, 0.029, -0.225).
    [-1.998, 0.858, -2.282, -0.614, 1.223, -2.090, 2.965],
    # Lowest pose using the visually confirmed -Z7 tool direction. The public
    # FK estimate is (0, 0, -0.104) for an 18 cm tool, and lower for the full
    # gripper/finger length.
    [-2.197, -0.470, 0.0, -0.469, -2.408, 0.0, -0.748],
]

pose_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
env = make_env()
state, _ = env.reset(seed=1)
robot = state.get_object_from_name("robot")
rods = list(state.get_objects(env.observation_space.get_type("mujoco_movable_object")))


def get(obj, feature):
    return float(state.get(obj, feature))


def xyz(obj):
    return np.array([get(obj, feature) for feature in ("x", "y", "z")])


initial = {rod.name: xyz(rod) for rod in rods}
target = np.asarray(POSES[pose_index])

# Reach the pose while open. Shortest-angle control is appropriate for the odd,
# continuous Gen3 joints and harmless for these in-range even-joint targets.
for step in range(150):
    q = np.array([get(robot, f"pos_arm_joint{i}") for i in range(1, 8)])
    error = target - q
    # Joints 1, 3, 5, and 7 are continuous. The even joints have hard stops,
    # so wrapping their errors can incorrectly drive them into the other stop.
    error[[0, 2, 4, 6]] = (
        error[[0, 2, 4, 6]] + np.pi
    ) % (2 * np.pi) - np.pi
    action = np.zeros(11, dtype=np.float32)
    action[3:10] = np.clip(0.8 * error, -0.1, 0.1)
    action[10] = 1.0
    state, _, _, _, _ = env.step(action)
    if np.max(np.abs(error)) < 0.025:
        break

print("pose", pose_index, "steps", step, "q", np.round(q, 3), flush=True)

# Scan across the rods. At every point explicitly open then close, since grasp
# attachment appears to be edge-triggered. Lateral rows cover FK/model error.
sample = 0
for y_offset in (-0.08, 0.0, 0.08):
    rod = rods[0]
    desired_y = get(rod, "y") + y_offset
    for x in np.linspace(-0.25, 0.75, 15):
        for _ in range(4):
            error = np.array([
                x - get(robot, "pos_base_x"),
                desired_y - get(robot, "pos_base_y"),
            ])
            if np.max(np.abs(error)) < 0.018:
                break
            action = np.zeros(11, dtype=np.float32)
            action[:2] = np.clip(error / 0.87, -0.1, 0.1)
            action[10] = 1.0
            state, _, _, _, _ = env.step(action)
        action = np.zeros(11, dtype=np.float32)
        action[10] = 0.0
        state, _, _, _, _ = env.step(action)
        sample += 1
        moved = {r.name: np.linalg.norm(xyz(r) - initial[r.name]) for r in rods}
        if max(moved.values()) > 0.003:
            print("HIT", sample, "base", get(robot, "pos_base_x"),
                  get(robot, "pos_base_y"), "moves", moved, flush=True)
            env.close()
            raise SystemExit

print("NO HIT", sample, flush=True)
env.close()
