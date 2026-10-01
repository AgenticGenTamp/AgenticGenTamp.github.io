from env_client import make_env
import numpy as np
env = make_env()
for seed in range(3):
    obs, info = env.reset(seed=seed)
    print("seed", seed, "info", info)
    for n in obs.get_object_names():
        o = obs.get_object_from_name(n)
        print(n, o.type if hasattr(o,'type') else '')
    r = obs.get_objects(env.observation_space.get_type('mujoco_tidybot_robot'))[0]
    print({f: round(obs.get(r,f),3) for f in ['pos_base_x','pos_base_y','pos_base_rot','pos_arm_joint1','pos_arm_joint2','pos_arm_joint3','pos_arm_joint4','pos_arm_joint5','pos_arm_joint6','pos_arm_joint7','pos_gripper']})
    for c in obs.get_objects(env.observation_space.get_type('mujoco_movable_object')):
        print(c.name, [round(obs.get(c,f),3) for f in ['x','y','z','qw','qx','qy','qz','bb_x','bb_y','bb_z']])
print(env.max_steps)
env.close()
