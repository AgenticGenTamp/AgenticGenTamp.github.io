import numpy as np
from env_client import make_env


def dump(seed):
    env = make_env()
    obs, info = env.reset(seed=seed)
    print("SEED", seed, "max", env.max_steps, "info", info)
    print("names", obs.get_object_names())
    for name in obs.get_object_names():
        obj = obs.get_object_from_name(name)
        typ = getattr(obj, "type", None)
        print(name, typ)
        if name == "robot":
            fs = ("pos_base_x pos_base_y pos_base_rot pos_arm_joint1 pos_arm_joint2 "
                  "pos_arm_joint3 pos_arm_joint4 pos_arm_joint5 pos_arm_joint6 "
                  "pos_arm_joint7 pos_gripper vel_base_x vel_base_y vel_base_rot "
                  "vel_arm_joint1 vel_arm_joint2 vel_arm_joint3 vel_arm_joint4 "
                  "vel_arm_joint5 vel_arm_joint6 vel_arm_joint7 vel_gripper").split()
        else:
            fs = "x y z qw qx qy qz vx vy vz wx wy wz bb_x bb_y bb_z".split()
        print({f: round(float(obs.get(obj, f)), 4) for f in fs})
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    obs2, r, term, trunc, info2 = env.step(a)
    print("zero step", r, term, trunc, info2)
    env.close()


if __name__ == "__main__":
    for seed in range(5):
        dump(seed)
