import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
def make_env(N):
    return VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:ScoopPour3DEnv",count_kwarg="num_objects",count_object_prefix="cube_",constant_object_env_kwargs={'scene_bg': False},design_counts=[N],eval_counts=[N])
def robot_q(s):
    r=s.get_object_from_name('robot')
    base=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
    arm=np.array([s.get(r,'pos_arm_joint%d'%i) for i in range(1,8)])
    return base, arm
def obj(s,n):
    o=s.get_object_from_name(n); return np.array([s.get(o,f) for f in ['x','y','z','qw','qx','qy','qz']])
def act(base,arm,s,bt,at,g):
    a=np.zeros(11,dtype=np.float32)
    db=bt-base; db[2]=(db[2]+np.pi)%(2*np.pi)-np.pi
    a[:3]=np.clip(db,-0.1,0.1); 
    da=(at-arm); 
    a[3:10]=np.clip(da,-0.1,0.1); a[10]=g
    return a
