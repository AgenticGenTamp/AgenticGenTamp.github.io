import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[4], eval_counts=[4])
s,info=env.reset(seed=0, options={'object_count': 4})
oc = env._current_backend._object_centric_env
sim = oc._robot_env.sim; m=sim.model.mj_model; d=sim.data.mj_data
import mujoco
lp=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_left_pad'); rp=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_right_pad')
print(lp,rp,[m.body(i).name for i in range(m.nbody) if 'pad' in m.body(i).name])
for a in [0,0.2,0.4,0.5,0.6,0.7,0.8,1.0,0.0]:
    act=np.zeros(11); act[10]=a
    for i in range(3): s,*_=env.step(act)
    r=s.get_object_from_name('robot')
    # pad geoms
    print(a, round(s.get(r,'pos_gripper'),3), np.linalg.norm(d.xpos[lp]-d.xpos[rp]).round(4))
