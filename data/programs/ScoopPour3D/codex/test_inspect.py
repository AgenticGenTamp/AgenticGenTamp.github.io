"""Small live-environment diagnostics; not imported by the policy."""

import numpy as np
from env_client import make_env


def snapshot(seed=0):
    env = make_env()
    state, info = env.reset(seed=seed)
    print("max_steps", env.max_steps, "info", info)
    print("action", env.action_space.low, env.action_space.high)
    print("names", state.get_object_names())
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        print("object", name, "type", getattr(obj, "type", None))
    movable_type = env.observation_space.get_type("mujoco_movable_object")
    robot_type = env.observation_space.get_type("mujoco_tidybot_robot")
    for obj in state.get_objects(movable_type):
        name = getattr(obj, "name", str(obj))
        vals = [state.get(obj, f) for f in ("x", "y", "z", "qw", "qx", "qy", "qz", "bb_x", "bb_y", "bb_z")]
        print(name, np.round(vals, 4).tolist())
    robot = state.get_objects(robot_type)[0]
    fs = ["pos_base_x", "pos_base_y", "pos_base_rot"] + [f"pos_arm_joint{i}" for i in range(1, 8)] + ["pos_gripper"]
    print("robot", {f: round(float(state.get(robot, f)), 4) for f in fs})
    env.close()


if __name__ == "__main__":
    snapshot()
