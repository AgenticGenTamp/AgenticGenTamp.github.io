import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
def make_env():
    return VariableObjectCountEnv(
    constant_object_env_path="robocode.environments.generalized_dynamic2d:CountParameterizedDynScoopPour2DEnv",
    count_kwarg="num_small_objects", count_object_prefix="small_",
    design_counts=[1,2,3,4,5,6,8], eval_counts=[1,2,3,4,5,6,8,10])
def rob(s):
    r=[o for o in s.get_object_names() if s.get_object_from_name(o).type.name=='kin_robot'][0]
    o=s.get_object_from_name(r)
    return {f: float(s.get(o,f)) for f in ['x','y','theta','arm_joint','finger_gap']}
def hook(s):
    o=s.get_object_from_name('hook')
    return {f: float(s.get(o,f)) for f in ['x','y','theta','held']}
