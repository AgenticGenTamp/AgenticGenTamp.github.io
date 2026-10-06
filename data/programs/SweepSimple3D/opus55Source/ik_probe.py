import numpy as np, mujoco
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SweepSimple3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube_", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[1,5,10,50], eval_counts=[1,5,10,50])
s,_=env.reset(seed=0, options={'object_count':5})
oc = env._current_backend._object_centric_env
m = oc._robot_env.sim.model.mj_model; d = oc._robot_env.sim.data.mj_data
for j in range(m.njnt):
    n=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    if n and ('robot' in n or 'joint' in n): print(j,n,m.jnt_type[j],m.jnt_qposadr[j], d.qpos[m.jnt_qposadr[j]])
for i in range(m.nsite): print('site',mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i), d.site_xpos[i])
for i in range(m.nu): print('act',mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
mujoco.mj_forward(m,d)
for g in range(m.ngeom):
    b = m.geom_bodyid[g]; bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ''
    if not bn.startswith('robot'): continue
    if m.geom_contype[g]==0 and m.geom_conaffinity[g]==0: continue
    c = d.geom_xpos[g]; R=d.geom_xmat[g].reshape(3,3); aabb = m.geom_aabb[g]
    cen = c + R@aabb[:3]; half = np.abs(R)@aabb[3:]
    print('G',bn, m.geom_type[g], m.geom_contype[g], m.geom_conaffinity[g], 'zmin %.3f zmax %.3f'%((cen-half)[2],(cen+half)[2]), 'fr',m.geom_friction[g][0])
for g in range(m.ngeom):
    bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ''
    if bn.startswith('wiper') or bn.startswith('cube_0'): print('W',bn,m.geom_type[g],m.geom_size[g],m.geom_pos[g],m.geom_friction[g],m.geom_contype[g],m.geom_conaffinity[g])
