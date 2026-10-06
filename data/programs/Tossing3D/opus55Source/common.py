import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
def make_env(counts=(1,2)):
    return VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:Tossing3DEnv", count_kwarg="num_objects", count_object_prefix="cube_", constant_object_env_kwargs={'scene_bg': False}, design_counts=list(counts), eval_counts=list(counts), base_steps=400, steps_per_object=400)
def robot(s):
    r = s.get_object_from_name('robot')
    base = np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
    q = np.array([s.get(r,f'pos_arm_joint{i}') for i in range(1,8)])
    return base, q
def obj(s, name):
    o = s.get_object_from_name(name)
    return np.array([s.get(o,f) for f in ['x','y','z','qw','qx','qy','qz','vx','vy','vz']])
