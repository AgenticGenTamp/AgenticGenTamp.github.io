import numpy as np, mujoco
m=mujoco.MjModel.from_xml_path("/robocode/third-party/kindergarden/src/kinder/envs/dynamic3d/models/stanford_tidybot/tidybot.xml")
d=mujoco.MjData(m); mujoco.mj_forward(m,d)
for g in [2,3]:
    print(g, m.geom_aabb[g], d.geom_xpos[g])
    mid=m.geom_dataid[g]
    if mid>=0:
        va=m.mesh_vertadr[mid]; vn=m.mesh_vertnum[mid]
        v=m.mesh_vert[va:va+vn]
        R=d.geom_xmat[g].reshape(3,3); w=v@R.T+d.geom_xpos[g]
        print(" world aabb",w.min(0).round(3),w.max(0).round(3))
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:Shelf3DEnv",count_kwarg="num_objects",count_object_prefix="cube",constant_object_env_kwargs={'scene_bg': False},design_counts=[1],eval_counts=[1])
s,_=env.reset(seed=0,options={'object_count':1})
r=s.get_object_from_name('robot')
def q(s): return np.array([s.get(r,f'pos_arm_joint{i}') for i in range(1,8)])
q0=q(s)
for k in range(30):
    a=np.zeros(11,dtype=np.float32); a[4]=0.1; a[6]=0.1; a[9]=0.1;a[0]=0.1;a[2]=0.1
    s,*_=env.step(a)
    if k%3==0: print(k,(q(s)-q0).round(3), round(s.get(r,'pos_base_x'),3), round(s.get(r,'pos_base_rot'),3))
