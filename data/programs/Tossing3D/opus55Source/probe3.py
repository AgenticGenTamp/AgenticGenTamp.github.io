import numpy as np, mujoco
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:Tossing3DEnv", count_kwarg="num_objects", count_object_prefix="cube_", constant_object_env_kwargs={'scene_bg': False}, design_counts=[1], eval_counts=[1])
s,_=env.reset(seed=0, options={'object_count':1})
sim = env._current_backend._object_centric_env._robot_env.sim
m = sim.model.mj_model
print('nq',m.nq,'nu',m.nu, 'timestep', m.opt.timestep)
for i in range(m.njnt): print(i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i), m.jnt_qposadr[i], m.jnt_type[i])
for i in range(m.nsite): print('site', mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i))
for i in range(m.nu): print('act', mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
