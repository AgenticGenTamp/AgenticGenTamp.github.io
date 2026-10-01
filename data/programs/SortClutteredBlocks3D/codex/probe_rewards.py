"""Small black-box survey for SortClutteredBlocks3DEnv."""
from env_client import make_env
import numpy as np


MOVABLE_FEATURES = (
    "x y z qw qx qy qz vx vy vz wx wy wz bb_x bb_y bb_z".split()
)
ROBOT_FEATURES = (
    "pos_base_x pos_base_y pos_base_rot pos_arm_joint1 pos_arm_joint2 "
    "pos_arm_joint3 pos_arm_joint4 pos_arm_joint5 pos_arm_joint6 "
    "pos_arm_joint7 pos_gripper"
).split()


def vals(state, name, features):
    obj = state.get_object_from_name(name)
    return [float(state.get(obj, f)) for f in features]


def summarize(seed, object_count=None, steps=2):
    env = make_env()
    try:
        kwargs = {"seed": seed}
        if object_count is not None:
            kwargs["options"] = {"object_count": object_count}
        state, info = env.reset(**kwargs)
        names = sorted(state.get_object_names())
        cubes = [n for n in names if n.startswith("cube")]
        print("RESET", seed, object_count, "info", info, "names", names)
        for n in [x for x in names if x.startswith("bin_") or x.startswith("cube")]:
            v = vals(state, n, MOVABLE_FEATURES)
            print("OBJ", n, "xyz", np.round(v[:3], 5), "quat", np.round(v[3:7], 4),
                  "bb", np.round(v[-3:], 5))
        print("ROBOT", np.round(vals(state, "robot", ROBOT_FEATURES), 5))
        for label, action in [
            ("zero_open", np.r_[np.zeros(10), 1.0]),
            ("zero_closed", np.zeros(11)),
        ][:steps]:
            state, reward, term, trunc, inf = env.step(action.astype(np.float32))
            print("STEP", label, "reward", reward, "term", term, "trunc", trunc,
                  "info", inf, "robot", np.round(vals(state, "robot", ROBOT_FEATURES), 4))
            for n in cubes:
                print(" CUBE", n, np.round(vals(state, n, MOVABLE_FEATURES)[:3], 5))
    except Exception as exc:
        print("ERROR", seed, object_count, type(exc).__name__, str(exc))
    finally:
        env.close()


if __name__ == "__main__":
    for count in (1, 2, 4, 8):
        summarize(100 + count, count, steps=1)
    for seed in (0, 1, 2, 3, 42):
        summarize(seed, steps=1)
