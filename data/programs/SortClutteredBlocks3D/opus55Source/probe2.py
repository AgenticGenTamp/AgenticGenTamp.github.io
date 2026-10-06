import numpy as np, mujoco
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[4], eval_counts=[4])
s,info=env.reset(seed=0, options={'object_count': 4})
oc = env._current_backend._object_centric_env
print(type(oc))
sim = oc._robot_env.sim
m=sim.model.mj_model; d=sim.data.mj_data
print([m.site(i).name for i in range(m.nsite)][:20])
sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'robot_pinch_site')
print("live pinch", d.site_xpos[sid], d.site_xmat[sid].reshape(3,3))
P="/robocode/third-party/kindergarden/src/kinder/envs/dynamic3d/models/stanford_tidybot/tidybot.xml"
M=mujoco.MjModel.from_xml_path(P); D=mujoco.MjData(M)
print(M.nq, [M.joint(i).name for i in range(M.njnt)])
r=s.get_object_from_name('robot')
q=[s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+[f'pos_arm_joint{i}' for i in range(1,8)]]
for i,n in enumerate(['joint_x','joint_y','joint_th']+[f'joint_{i}' for i in range(1,8)]):
    D.qpos[M.jnt_qposadr[M.joint(n).id]]=q[i]
mujoco.mj_kinematics(M,D)
sid2=mujoco.mj_name2id(M,mujoco.mjtObj.mjOBJ_SITE,'pinch_site')
print("mine pinch", D.site_xpos[sid2], D.site_xmat[sid2].reshape(3,3))
