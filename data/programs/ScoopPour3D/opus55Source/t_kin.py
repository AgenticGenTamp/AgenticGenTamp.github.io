import numpy as np, mujoco
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
N=10
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:ScoopPour3DEnv",count_kwarg="num_objects",count_object_prefix="cube_",constant_object_env_kwargs={'scene_bg': False},design_counts=[N],eval_counts=[N])
s,info=env.reset(seed=0, options={'object_count':N})
oc=env._current_backend._object_centric_env
m=oc._robot_env.sim.model.mj_model; d=oc._robot_env.sim.data.mj_data
sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'robot_pinch_site')
print('site', d.site_xpos[sid], d.site_xmat[sid].reshape(3,3))
for j in ['robot_joint_x','robot_joint_y','robot_joint_th']+['robot_joint_%d'%i for i in range(1,8)]:
    jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,j); print(j, m.jnt_qposadr[jid], m.jnt_dofadr[jid], m.jnt_range[jid])
for b in ['robot_left_pad','robot_right_pad','robot_base_link']:
    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,b); print(b, d.xpos[bid])
r=s.get_object_from_name('robot'); print([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
# island geoms
for i in range(m.nbody):
    nm=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i)
    if 'island' in nm or 'counter' in nm: print(i,nm,d.xpos[i])
# find geoms with top near z 0.46
for g in range(m.ngeom):
    b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g])
    if 'island' in b and m.geom_contype[g]:
        print(g,b,m.geom_type[g],np.round(d.geom_xpos[g],3),np.round(m.geom_size[g],3), np.round(m.geom_aabb[g],3))
