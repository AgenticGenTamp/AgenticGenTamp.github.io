import numpy as np, time
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env=KinderGeom3DEnv("kinder/BalanceBeam3D-o3-v0")
ke=env._kinder_env; oc=ke._object_centric_env; re=oc._robot_env
d=re.sim.data.mj_data
obs,_=env.reset(seed=0)
rng=np.random.default_rng(0)
for _ in range(5): env.step(rng.uniform(-0.05,0.05,11).astype(np.float32))
s=env.get_state(); acts=[rng.uniform(-0.05,0.05,11).astype(np.float32) for _ in range(10)]
def run():
    for a in acts: o,*_=env.step(a)
    return o
import mujoco; m=re.sim.model.mj_model; spec=mujoco.mjtState.mjSTATE_INTEGRATION; snap=np.empty(mujoco.mj_stateSize(m,spec)); mujoco.mj_getState(m,d,snap,spec)
o1=run()
oc._set_state(ke.observation_space.devectorize(s)); o2=run()
print("direct _set_state replay diff", np.abs(o1-o2).max())
t=time.time()
mujoco.mj_setState(m,d,snap,spec)
import mujoco; mujoco.mj_forward(re.sim.model.mj_model,d); oc._current_state=oc._get_object_centric_state()
print("raw restore time", time.time()-t)
o3=run(); dd=np.abs(o1-o3); print("raw restore replay diff", dd.max(), np.argsort(dd)[-4:], np.sort(dd)[-4:]); dd=np.abs(o1-o2); print("direct", np.argsort(dd)[-3:], np.sort(dd)[-3:])
