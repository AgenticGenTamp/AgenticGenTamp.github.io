import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[4], eval_counts=[4])
s,info=env.reset(seed=0, options={'object_count': 4})
oc = env._current_backend._object_centric_env
sim = oc._robot_env.sim; d=sim.data.mj_data; m=sim.model.mj_model
def rq(s):
    r=s.get_object_from_name('robot')
    a=np.array([s.get(r,f'pos_arm_joint{i}') for i in range(1,8)])
    v=np.array([s.get(r,f'vel_arm_joint{i}') for i in range(1,8)])
    return a,v
a0,_=rq(s)
for k in range(8):
    act=np.zeros(11); act[3:10]=[0.3,0.3,0,0,0,0,0.3] if k<3 else 0
    if k>=3: act[3:10]=1.5*(a0+np.array([0.3*3,0.3*3,0,0,0,0,0.9])*0-a0)
    s,*_=env.step(act)
    a,v=rq(s)
    print(k, (a-a0).round(3)[[0,1,6]], v.round(3)[[0,1,6]])
