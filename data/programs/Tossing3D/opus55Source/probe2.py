import time, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:Tossing3DEnv", count_kwarg="num_objects", count_object_prefix="cube_", constant_object_env_kwargs={'scene_bg': False}, design_counts=[1], eval_counts=[1])
s,_=env.reset(seed=0, options={'object_count':1})
inner = env._current_backend._object_centric_env
print(type(inner), [a for a in dir(inner) if not a.startswith('__')][:80])
t=time.time()
for i in range(20): env.step(np.zeros(18))
print('step time', (time.time()-t)/20)
re_ = inner._robot_env
print(type(re_), [a for a in dir(re_) if not a.startswith('__')])
sim = re_.sim
print(type(sim), [a for a in dir(sim) if not a.startswith('__')])
import mujoco
m = sim.model if hasattr(sim,'model') else None
print(type(m))
m = getattr(m,'_model',m)
print('nq',m.nq,'nu',m.nu)
for i in range(m.njnt): print(i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i), m.jnt_qposadr[i], m.jnt_type[i])
for i in range(m.nsite): print('site', mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i))
for i in range(m.nu): print('act', mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
