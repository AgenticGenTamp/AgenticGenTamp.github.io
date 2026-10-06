import numpy as np, mujoco, time
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
k=env._kinder_env; oc=k._object_centric_env; re=oc._robot_env
m=re.sim.model.mj_model; d=re.sim.data.mj_data
env.reset(seed=0)
print("arm qpos idx", re.qpos["arm"].indices if hasattr(re.qpos["arm"],'indices') else None, "qvel", re._arm_qvel_start, re._arm_qvel_end, "base qpos idx", getattr(re.qpos["base"],'indices',None))
sid=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot_pinch_site")
print("pinch site pos", d.site_xpos[sid], "xmat", np.round(d.site_xmat[sid].reshape(3,3),3))
# Determinism: oc._set_state
s0=env.get_state(); st=k.observation_space.devectorize(s0)
rng=np.random.default_rng(0); acts=rng.uniform(-0.05,0.05,(15,11)); acts[:,10]=rng.uniform(0,1,15)
def roll():
    for a in acts: o,*_=env.step(a)
    return o
o1=roll(); oc._set_state(st); env._current_obs=k.observation_space.vectorize(oc._get_state()); o2=roll()
print("oc._set_state then same actions maxdiff", np.abs(o1-o2).max())
# full mujoco state snapshot
spec=mujoco.mjtState.mjSTATE_INTEGRATION
env.reset(seed=0); buf=np.empty(mujoco.mj_stateSize(m,spec)); mujoco.mj_getState(m,d,buf,spec)
o1=roll(); mujoco.mj_setState(m,d,buf,spec); mujoco.mj_forward(m,d); o2=roll()
print("mj_getState/setState then same actions maxdiff", np.abs(o1-o2).max())
# Simple DLS IK on copy
env.reset(seed=0)
d2=mujoco.MjData(m); d2.qpos[:]=d.qpos; mujoco.mj_forward(m,d2)
qa=np.array(re.qpos["arm"].indices) if hasattr(re.qpos["arm"],'indices') else None
va=np.arange(re._arm_qvel_start, re._arm_qvel_end)
target=d.site_xpos[sid]+np.array([0.1,0,-0.1])
t=time.time()
for it in range(100):
    err=target-d2.site_xpos[sid]
    if np.linalg.norm(err)<1e-4: break
    J=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d2,J,None,sid); J=J[:,va]
    dq=J.T@np.linalg.solve(J@J.T+1e-4*np.eye(3),err)
    d2.qpos[qa]+=dq; mujoco.mj_kinematics(m,d2)
print("IK iters",it,"err",np.linalg.norm(target-d2.site_xpos[sid]),"time",time.time()-t)
