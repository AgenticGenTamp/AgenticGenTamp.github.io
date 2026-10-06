import time, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.obstruction3d:Obstruction3DEnv", count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[1,2,3], eval_counts=[1,2,3])
s, info = env.reset(seed=0, options={'object_count': 3})
print(type(s), type(env.unwrapped) if hasattr(env,'unwrapped') else None)
t=time.time()
from kinder.envs.kinematic3d.obstruction3d import Obstruction3DEnv, ObjectCentricObstruction3DEnv
import kinder.envs.kinematic3d.obstruction3d as m
print([n for n in dir(m) if 'Env' in n])
