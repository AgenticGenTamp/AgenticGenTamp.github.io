import numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
k = env._kinder_env; oc = k._object_centric_env
md = k.metadata
print(md.get('action_space_description','')[:1500])
print(md.get('reward_description','')[:800])
for seed in range(5):
    obs,_ = env.reset(seed=seed)
    s = oc._current_state
    print(f"--- seed {seed}")
    for o in s:
        if o.name.startswith("cube") : print(o.name, np.round([s.get(o,f) for f in ["x","y","z"]],4))
        if o.name.startswith("wiper"): print(o.name, np.round([s.get(o,f) for f in ["x","y","z","qw","qx","qy","qz","bb_x","bb_y","bb_z"]],4))
        if o.name=="robot": print("robot base", np.round([s.get(o,f) for f in ["pos_base_x","pos_base_y","pos_base_rot"]],4), "arm", np.round([s.get(o,f"pos_arm_joint{i}") for i in range(1,8)],3), "grip", round(s.get(o,"pos_gripper"),3))
        if o.name.startswith("kitchen_island_drawer"): print(o.name, round(s.get(o,"pos"),5), end="; ")
    print()
