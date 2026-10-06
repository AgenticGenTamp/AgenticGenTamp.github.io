import numpy as np, mujoco
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
N=10
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:ScoopPour3DEnv",count_kwarg="num_objects",count_object_prefix="cube_",constant_object_env_kwargs={'scene_bg': False},design_counts=[N],eval_counts=[N])
s,info=env.reset(seed=0, options={'object_count':N})
be = env._current_backend if hasattr(env,'_current_backend') else None
print(type(be))
oc = be._object_centric_env if hasattr(be,'_object_centric_env') else None
print(type(oc))
m = oc._robot_env.sim.model.mj_model
for i in range(m.nbody):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
    if any(k in nm for k in ['bin','scoop','cube_0','robot']): print(i, nm, round(m.body_mass[i],3), round(m.body_subtreemass[i],3))
print('ngeom',m.ngeom)
for i in range(m.ngeom):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ''
    b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i])
    if any(k in b for k in ['bin_yellow','scoop','cube_0','finger','pad','robot_bracelet','gripper']): print(i,nm,b,m.geom_type[i],np.round(m.geom_size[i],3),np.round(m.geom_pos[i],3), m.geom_friction[i], m.geom_contype[i], m.geom_conaffinity[i])
print(m.opt.timestep, m.opt.gravity)
