"""Scan low-dimensional joint families for a grasp proximity signal."""
from env_client import make_env
import numpy as np


def get(s, o, f):
    return float(s.get(o, f))


def act(env, s, values):
    a = np.zeros(11, dtype=np.float32)
    for i, x in values.items():
        a[i] = x
    return env.step(a)[0]


def run(joint_axis, direction, seed=0, steps=32):
    env = make_env()
    s, _ = env.reset(seed=seed)
    part = s.get_object_from_name("part0")
    rack = s.get_object_from_name("rack")
    robot = s.get_object_from_name("robot")
    target_y = get(s, part, "pose_y")
    target_x = get(s, robot, "pos_base_x") + get(s, part, "pose_x") - get(s, rack, "pose_x")
    # Align the base under part0 and explicitly open.
    for _ in range(3):
        dy = target_y - get(s, robot, "pos_base_y")
        dx = target_x - get(s, robot, "pos_base_x")
        s = act(env, s, {0: max(-0.2, min(0.2, dx)),
                         1: max(-0.2, min(0.2, dy)), 10: 1.0})
    for k in range(steps):
        s = act(env, s, {joint_axis: 0.2 * direction, 10: 1.0})
        before = [get(s, robot, f"joint_{i}") for i in range(1, 8)]
        s = act(env, s, {10: -1.0})
        grasp = get(s, robot, "grasp_active")
        if grasp:
            tf = [get(s, robot, f"grasp_tf_{q}") for q in ("x", "y", "z")]
            print("SUCCESS", joint_axis, direction, k, before, tf)
            env.close()
            return
        s = act(env, s, {10: 1.0})
        after = [get(s, robot, f"joint_{i}") for i in range(1, 8)]
        if k and max(abs(a-b) for a, b in zip(before, after)) < 1e-8:
            pass
        print("scan", joint_axis, direction, k, before)
    env.close()


if __name__ == "__main__":
    # Actions 4/6/8 correspond to arm joints 2/4/6.
    for axis in (4, 6, 8):
        for sign in (-1, 1):
            run(axis, sign)
