from run_test import *
import mujoco
s, info = env.reset(seed=0, options={'object_count': 1})
oc = env._current_backend._object_centric_env
m = oc._robot_env.sim.model.mj_model
for i in range(m.nu):
    print(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i), m.actuator_gainprm[i][:3], m.actuator_biasprm[i][:3], m.actuator_forcerange[i], m.actuator_ctrlrange[i], m.actuator_trntype[i])
for j in range(m.njnt):
    n=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    if n.startswith('robot_joint_'): print(n, 'armature', m.dof_armature[m.jnt_dofadr[j]], 'damping', m.dof_damping[m.jnt_dofadr[j]], 'frictionloss', m.dof_frictionloss[m.jnt_dofadr[j]])
