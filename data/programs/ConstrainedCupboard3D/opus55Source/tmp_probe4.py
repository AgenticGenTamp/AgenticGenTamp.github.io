import numpy as np, mujoco
np.set_printoptions(precision=4, suppress=True, linewidth=220)
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:ConstrainedCupboard3DEnv", count_kwarg="num_objects", count_object_prefix="cuboid_", constant_object_env_kwargs={'scene_bg': 'mimiclabs-lab2'}, design_counts=[1,2,3], eval_counts=[1,2,3])
state, info = env.reset(seed=0, options={'object_count': 2})
re = env._current_backend._object_centric_env._robot_env
m, d = re.sim.model.mj_model, re.sim.data.mj_data
J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
for n in ["robot_joint_x","robot_joint_y","robot_joint_th"]+[f"robot_joint_{i}" for i in range(1,8)]:
    j=J(n); print(n, "qadr", m.jnt_qposadr[j], "dadr", m.jnt_dofadr[j], "limited", m.jnt_limited[j], m.jnt_range[j])
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot_pinch_site")
bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot_base_link"); ab = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot_gen3/base_link")
print("base_link xpos", d.xpos[bb], "arm base xpos", d.xpos[ab], "pinch", d.site_xpos[sid])
print("base qpos", re.qpos['base'])
# Separate MjData FK/IK
d2 = mujoco.MjData(m)
d2.qpos[:] = d.qpos
arm_q = [m.jnt_qposadr[J(f"robot_joint_{i}")] for i in range(1,8)]
arm_v = [m.jnt_dofadr[J(f"robot_joint_{i}")] for i in range(1,8)]
mujoco.mj_kinematics(m, d2); mujoco.mj_comPos(m, d2)
print("FK copy pinch", d2.site_xpos[sid], "matches", np.allclose(d2.site_xpos[sid], d.site_xpos[sid], atol=1e-6))
target = d2.site_xpos[sid] + np.array([0.15, 0.1, -0.2])
for it in range(100):
    mujoco.mj_kinematics(m, d2); mujoco.mj_comPos(m, d2)
    err = target - d2.site_xpos[sid]
    if np.linalg.norm(err) < 1e-4: break
    jp = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d2, jp, None, sid)
    Ja = jp[:, arm_v]
    dq = Ja.T @ np.linalg.solve(Ja@Ja.T + 1e-4*np.eye(3), err)
    d2.qpos[arm_q] += dq
print("IK iters", it, "err", np.linalg.norm(err), "q", d2.qpos[arm_q])
# drive real env to IK solution
qt = d2.qpos[arm_q].copy()
for t in range(30):
    a = np.zeros(11); a[3:10] = qt - re.qpos['arm']; env.step(a)
print("after 30 steps pinch", d.site_xpos[sid], "target", target, "dist", np.linalg.norm(d.site_xpos[sid]-target))
