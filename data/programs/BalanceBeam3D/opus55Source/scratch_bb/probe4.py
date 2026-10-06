import time, numpy as np, mujoco
np.set_printoptions(precision=4, suppress=True)
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env=KinderGeom3DEnv("kinder/BalanceBeam3D-o3-v0")
ke=env._kinder_env; oc=ke._object_centric_env; re=oc._robot_env
obs,_=env.reset(seed=0)
st=obs.copy()
env.set_state(st); o=env.get_state()
print("wrapper set_state: LB before",st[0:3],"after",o[0:3], "seesaw before",st[38:40],"after",o[38:40])
# direct _set_state
env.reset(seed=0); a=np.zeros(11,np.float32)
for _ in range(3): env.step(a)
s0=env.get_state()
t=time.time(); oc._set_state(ke.observation_space.devectorize(s0)); print("direct _set_state time", time.time()-t)
o1=ke.observation_space.vectorize(oc._get_state()); print("direct roundtrip diff", np.abs(o1-s0).max())
# controller tracking test
env.reset(seed=0)
m=re.sim.model.mj_model; d=re.sim.data.mj_data
sid=[i for i in range(m.nsite) if 'pinch' in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_SITE,i) or '')]
print("pinch sites", [mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_SITE,i) for i in sid], d.site_xpos[sid[0]])
o=env.get_state()
for k in range(6):
    a=np.zeros(11,np.float32); a[3]=0.1; a[0]=0.1; a[2]=0.1
    o2,r,*_=env.step(a)
    print(f"step{k} dbase {o2[16:19]-o[16:19]} djoint1 {o2[19]-o[19]:.4f} other joints drift {np.abs(o2[20:26]-o[20:26]).max():.4f} vel1 {o2[30]:.3f}")
    o=o2
for k in range(5):
    o2,*_=env.step(np.zeros(11,np.float32)); print(f" hold{k} dbase {o2[16:19]-o[16:19]} djoint1 {o2[19]-o[19]:.4f}"); o=o2
# gripper
for g in [1.0,1.0,1.0,0.0,0.0]:
    a=np.zeros(11,np.float32); a[10]=g; o2,*_=env.step(a)
    gq=re.qpos['gripper'] if 'gripper' in re.qpos else None
    print(" grip cmd",g,"obs pos_gripper",o2[26],"vel",o2[37],"gripper qpos",gq)
# big delta joint 4: what happens with delta beyond? clipped?
env.reset(seed=0); o=env.get_state()
a=np.zeros(11,np.float32); a[6]=0.5
o2,*_=env.step(a); print("delta 0.5 on joint4 -> moved", o2[22]-o[22])
print("qpos keys", list(re.qpos.keys()), {k:len(v) for k,v in re.qpos.items()})
print("timestep counter", re.timestep, "horizon", re.horizon)
