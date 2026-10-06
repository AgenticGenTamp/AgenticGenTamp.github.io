import time, numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
print("act low", env.action_space.low, "high", env.action_space.high)
print("obs shape", env.observation_space.shape)
ke = env._kinder_env
oc = ke._object_centric_env
print(type(ke), type(oc))
obs, _ = env.reset(seed=0)
st = oc._get_current_state()
for o in st:
    print(o.name, o.type.name, [(f, round(float(st.get(o,f)),3)) for f in st.type_features[o.type]])
re = oc._robot_env
print(type(re), re.sim.model.mj_model.opt.timestep)
m = re.sim.model.mj_model
print("sites:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)])
print("bodies:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)])
