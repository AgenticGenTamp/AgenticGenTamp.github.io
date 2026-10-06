import numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
k=env._kinder_env; oc=k._object_centric_env; re=oc._robot_env
m=re.sim.model.mj_model; d=re.sim.data.mj_data
env.reset(seed=0)
rng=np.random.default_rng(0); acts=rng.uniform(-0.05,0.05,(15,11)); acts[:,10]=rng.uniform(0,1,15)
def roll():
    for a in acts: o,*_=env.step(a)
    return o
spec=mujoco.mjtState.mjSTATE_INTEGRATION
buf=np.empty(mujoco.mj_stateSize(m,spec)); mujoco.mj_getState(m,d,buf,spec); w=d.qacc_warmstart.copy()
o1=roll(); mujoco.mj_setState(m,d,buf,spec); d.qacc_warmstart[:]=w; mujoco.mj_forward(m,d); o2=roll()
print("mj full restore (+warmstart) maxdiff", np.abs(o1-o2).max(), np.argmax(np.abs(o1-o2)))
mujoco.mj_setState(m,d,buf,spec); d.qacc_warmstart[:]=w; mujoco.mj_forward(m,d); o3=roll()
print("repeat restore o2 vs o3", np.abs(o3-o2).max())
