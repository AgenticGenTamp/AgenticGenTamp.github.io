"""Map seed-4 obstruction0 release after a lift / joint-1 arc / lower."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

import numpy as np

from env_client import make_env


def value(state, name, feature):
    obj = state.get_object_from_name(name)
    return float(state.get(obj, feature))


def run(final_q1):
    env = make_env()
    state, _ = env.reset(seed=4)
    name = "obstruction0"
    source = (value(state, name, "pose_x"), value(state, name, "pose_y"))
    # Same calibrated grasp used by the policy and test_arc.py.
    radius = 0.656
    base = (source[0] - radius, source[1])
    q_start = 0.0
    for _ in range(12):
        action = np.zeros(11, np.float32)
        for index, feature, goal in (
            (0, "pos_base_x", base[0]), (1, "pos_base_y", base[1]),
            (3, "joint_1", q_start), (4, "joint_2", 0.65),
            (6, "joint_4", -1.50), (8, "joint_6", -0.87),
        ):
            action[index] = np.clip(goal - value(state, "robot", feature), -0.2, 0.2)
        action[10] = -1.0
        state, _, _, _, _ = env.step(action)
        if value(state, "robot", "grasp_active") > 0.5:
            break
    grasped = value(state, "robot", "grasp_active") > 0.5
    if grasped:
        # Raise, swing on a horizontal circle, and lower to the grasp height.
        for feature, index, goal in (
            ("joint_6", 8, -0.77),
            ("joint_1", 3, final_q1),
            ("joint_6", 8, -0.87),
        ):
            for _ in range(12):
                action = np.zeros(11, np.float32)
                action[index] = np.clip(goal - value(state, "robot", feature), -0.2, 0.2)
                state, _, _, _, _ = env.step(action)
                if abs(value(state, "robot", feature) - goal) < 0.005:
                    break
        before = tuple(value(state, name, f"pose_{axis}") for axis in "xyz")
        action = np.zeros(11, np.float32)
        action[10] = 1.0
        state, _, _, _, _ = env.step(action)
        after = tuple(value(state, name, f"pose_{axis}") for axis in "xyz")
        released = value(state, "robot", "grasp_active") < 0.5
    else:
        before = after = (math.nan,) * 3
        released = False
    env.close()
    displacement = math.hypot(after[0] - source[0], after[1] - source[1])
    return final_q1, grasped, released, displacement, after, before


if __name__ == "__main__":
    endpoints = [round(x, 2) for x in np.arange(-1.20, 1.201, 0.20)]
    # The black-box server caps concurrent connections; four is reliable.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run, endpoint) for endpoint in endpoints]
        rows = sorted(future.result() for future in as_completed(futures))
    print("q1 grasp release planar_disp final_xyz pre_release_xyz")
    for row in rows:
        q1, grasped, released, displacement, after, before = row
        print(f"{q1:+.2f} {int(grasped)} {int(released)} {displacement:.4f} "
              f"{tuple(round(x, 4) for x in after)} {tuple(round(x, 4) for x in before)}")
