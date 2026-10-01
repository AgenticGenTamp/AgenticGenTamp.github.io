"""Empirical one-step kinematics probes for PR2Blocked (exploration only)."""

import math
import numpy as np

from env_client import make_env


ROBOT_FEATURES = [
    "base_x", "base_y", "base_rot",
    "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7",
    "gripper_opening", "grasp_active",
    "grasp_tf_x", "grasp_tf_y", "grasp_tf_z",
    "grasp_tf_qx", "grasp_tf_qy", "grasp_tf_qz", "grasp_tf_qw",
]


def robot_vec(state):
    obj = state.get_object_from_name("robot")
    return np.array([state.get(obj, f) for f in ROBOT_FEATURES])


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def axis_angle(q):
    if q[3] < 0:
        q = -q
    q = q / np.linalg.norm(q)
    ang = 2 * math.atan2(np.linalg.norm(q[:3]), q[3])
    axis = q[:3] / max(np.linalg.norm(q[:3]), 1e-12)
    return axis * ang


def main():
    env = make_env()
    try:
        for seed in [0, 1]:
            s, info = env.reset(seed=seed)
            v0 = robot_vec(s)
            print("SEED", seed, "objects", sorted(s.get_object_names()))
            for name in sorted(s.get_object_names()):
                if name == "robot":
                    continue
                obj = s.get_object_from_name(name)
                vals = []
                for f in ("pose_x", "pose_y", "pose_z", "half_extent_x", "half_extent_y", "half_extent_z"):
                    try:
                        vals.append(s.get(obj, f))
                    except ValueError:
                        vals.append(float("nan"))
                print("object", name, np.round(vals, 4))
            print("baseline base", np.round(v0[:3], 5), "joints", np.round(v0[3:10], 5))
            print("baseline tf", np.round(v0[12:15], 5), "q", np.round(v0[15:19], 5))
            for i in range(10):
                s, info = env.reset(seed=seed)
                b = robot_vec(s)
                a = np.zeros(11, dtype=np.float32)
                a[i] = 0.05
                s, rew, term, trunc, info = env.step(a)
                n = robot_vec(s)
                dp = n[12:15] - b[12:15]
                dq = qmul(n[15:19], qconj(b[15:19]))
                dr = axis_angle(dq)
                print(i, "state_delta", np.round(n[:10]-b[:10], 5),
                      "tf_dp", np.round(dp, 6), "tf_drot", np.round(dr, 6))
    finally:
        env.close()


if __name__ == "__main__":
    main()
