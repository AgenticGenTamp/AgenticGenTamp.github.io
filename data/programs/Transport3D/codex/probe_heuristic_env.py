from env_client import make_env
import numpy as np


def snap(state):
    out = {}
    for name in state.get_object_names():
        o = state.get_object_from_name(name)
        vals = {}
        for f in ("pose_x", "pose_y", "pose_z", "grasp_active",
                  "pos_base_x", "pos_base_y", "pos_base_rot",
                  "joint_1", "joint_2", "joint_3", "joint_4",
                  "joint_5", "joint_6", "joint_7", "finger_state",
                  "grasp_tf_x", "grasp_tf_y", "grasp_tf_z",
                  "half_extent_x", "half_extent_y", "half_extent_z"):
            try:
                vals[f] = round(float(state.get(o, f)), 4)
            except Exception:
                pass
        out[name] = vals
    return out


def one(seed, actions):
    env = make_env()
    s, info = env.reset(seed=seed)
    print("RESET", seed, "max", env.max_steps, snap(s), "info", info)
    for label, a, n in actions:
        for i in range(n):
            old = snap(s)
            s, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
            if i == 0 or i == n - 1 or term:
                print(label, i + 1, "rtt", r, term, trunc, "delta/now", snap(s), "info", info)
            if term or trunc:
                break
    env.close()


if __name__ == "__main__":
    z = [0.0] * 11
    one(0, [("zero", z, 1), ("base+x", [0.2,0,0,0,0,0,0,0,0,0,0], 1),
            ("base+y", [0,0.2,0,0,0,0,0,0,0,0,0], 1),
            ("all joints +", [0,0,0,.2,.2,.2,.2,.2,.2,.2,0], 1),
            ("close", [0,0,0,0,0,0,0,0,0,0,-1], 1)])
