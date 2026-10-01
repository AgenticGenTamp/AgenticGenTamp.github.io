"""Empirical control/kinematics probes for Obstruction3DEnv."""
import numpy as np

from env_client import make_env


ROBOT_FEATURES = (
    "pos_base_x", "pos_base_y", "pos_base_rot",
    "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7",
    "finger_state", "grasp_active",
    "grasp_tf_x", "grasp_tf_y", "grasp_tf_z",
    "grasp_tf_qx", "grasp_tf_qy", "grasp_tf_qz", "grasp_tf_qw",
)


def robot_values(state):
    robot = state.get_object_from_name("robot")
    return np.array([state.get(robot, f) for f in ROBOT_FEATURES], dtype=float)


def cuboids(state):
    ans = {}
    for name in state.get_object_names():
        if name == "robot":
            continue
        obj = state.get_object_from_name(name)
        try:
            ans[name] = np.array([state.get(obj, f) for f in (
                "pose_x", "pose_y", "pose_z", "half_extent_x", "half_extent_y", "half_extent_z",
                "grasp_active", "object_type")], dtype=float)
        except Exception:
            pass
    return ans


def one_step(seed, index, magnitude):
    env = make_env()
    before, _ = env.reset(seed=seed)
    vb = robot_values(before)
    a = np.zeros(11, dtype=np.float32)
    a[index] = magnitude
    after, reward, term, trunc, _ = env.step(a)
    va = robot_values(after)
    env.close()
    return vb, va, reward, term, trunc


def main():
    env = make_env()
    s, info = env.reset(seed=0)
    print("names", s.get_object_names())
    print("robot0", dict(zip(ROBOT_FEATURES, robot_values(s))))
    for name, vals in cuboids(s).items():
        print("object", name, vals)
    env.close()

    print("ONE-HOT DELTAS seed=0 magnitude=+0.2")
    for i in range(11):
        vb, va, rew, term, trunc = one_step(0, i, 0.2 if i < 10 else 1.0)
        changed = [(ROBOT_FEATURES[j], va[j] - vb[j], va[j])
                   for j in range(len(vb)) if abs(va[j] - vb[j]) > 1e-7]
        print(i, changed, rew, term, trunc)

    print("ONE-HOT DELTAS seed=0 magnitude=-0.2")
    for i in range(10):
        vb, va, *_ = one_step(0, i, -0.2)
        changed = [(ROBOT_FEATURES[j], va[j] - vb[j], va[j])
                   for j in range(len(vb)) if abs(va[j] - vb[j]) > 1e-7]
        print(i, changed)

    # Repeated positive commands expose scaling and joint/base limits.
    for i in range(10):
        env = make_env()
        state, _ = env.reset(seed=0)
        start = robot_values(state)
        a = np.zeros(11, dtype=np.float32)
        a[i] = 0.2
        for _ in range(20):
            state, *_ = env.step(a)
        end = robot_values(state)
        print("repeat20", i, "controlled", ROBOT_FEATURES[i], start[i], end[i],
              "ee_start", start[12:15], "ee_end", end[12:15])
        env.close()


if __name__ == "__main__":
    main()
