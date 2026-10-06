import numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
k=env._kinder_env; oc=k._object_centric_env; re=oc._robot_env
obs,_=env.reset(seed=0); s0=env.get_state()
for i in range(5): env.step(env.action_space.sample()*0)
env.set_state(s0); s1=env.get_state()
names = k.observation_space.feature_names if hasattr(k.observation_space,'feature_names') else None
d=np.abs(s1-s0); idx=np.where(d>1e-4)[0]
print("n diff", len(idx), idx[:40])
cs = k.observation_space.devectorize(s0); cs1=k.observation_space.devectorize(s1)
for o in cs:
    for f in cs.type_features[o.type]:
        a,b=cs.get(o,f),cs1.get(o,f)
        if abs(a-b)>1e-4: print(o.name,f,round(float(a),4),round(float(b),4))
# direct _set_state on object-centric env
oc.reset(seed=0); st=oc._get_state(); oc.step(np.zeros(11)); 
import copy
oc._set_state(st); st2=oc._get_state()
v0=k.observation_space.vectorize(st); v2=k.observation_space.vectorize(st2)
print("direct oc._set_state maxdiff", np.abs(v0-v2).max())
m=re.sim.model; mj=re.sim.data.mj_data
print("model attr:", [a for a in dir(m) if 'model' in a.lower() or a.startswith('mj')], type(mj))
import mujoco
mm = getattr(m,'_model',None) or getattr(m,'mj_model',None)
print("mj_model", type(mm))
print("sites", [mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(mm.nsite)][:40])
print("bodies robot", [mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(mm.nbody) if 'robot' in (mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, i) or '') ][:40])
print("qpos parts", {k_: len(v) for k_,v in re.qpos.items()}, "ctrl parts", {k_: len(v) for k_,v in re.ctrl.items()})
print("nq nv nu", mm.nq, mm.nv, mm.nu, "timestep", mm.opt.timestep)
