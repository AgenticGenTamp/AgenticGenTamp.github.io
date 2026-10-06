import time, numpy as np, gymnasium
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
print("act space", env.action_space.low, env.action_space.high, "obs", env.observation_space.shape)
print("gym spec max_episode_steps:", gymnasium.spec("kinder/SweepIntoDrawer3D-o5-v0").max_episode_steps)
k = env._kinder_env; oc = k._object_centric_env; re = oc._robot_env
print("chain types", type(k).__name__, type(oc).__name__, type(re).__name__, "horizon", re.horizon, "cf", re.control_frequency, "act_delta", re.act_delta)
obs,_ = env.reset(seed=0)
r = oc._current_state.get_objects(__import__('kinder.envs.dynamic3d.object_types',fromlist=['x']).MujocoTidyBotRobotObjectType)
st = oc._current_state; rob=list(r)[0]
print("robot feats", {f: round(float(st.get(rob,f)),4) for f in st.type_features[rob.type]})
print("qpos base", re.qpos["base"][:], "arm", np.round(re.qpos["arm"][:],4), "ctrl gripper", re.ctrl["gripper"][:])
m = re.sim.model._model if hasattr(re.sim.model,'_model') else None
print("sim.model attrs", [a for a in dir(re.sim.model) if not a.startswith('__')][:30])
print("sim.data attrs", [a for a in dir(re.sim.data) if not a.startswith('__')][:30])
# hold still with zeros
s0 = env.get_state()
b0 = np.array(re.qpos["base"][:]); a0 = np.array(re.qpos["arm"][:])
t=time.time()
for i in range(20): env.step(np.zeros(11))
dt=(time.time()-t)/20
print("time/step", dt, "after 20 zero steps base drift", np.array(re.qpos["base"][:])-b0, "arm drift", np.round(np.array(re.qpos["arm"][:])-a0,4))
# base +0.1 delta x one step
b1=np.array(re.qpos["base"][:]); a=np.zeros(11); a[0]=0.1
env.step(a); print("after +0.1 x delta, base moved", np.array(re.qpos["base"][:])-b1)
for i in range(3): env.step(np.zeros(11)); print("  then zero step base", np.array(re.qpos["base"][:])-b1)
# +1.0 (clipped?)
b1=np.array(re.qpos["base"][:]); a=np.zeros(11); a[0]=1.0
env.step(a); print("after +1.0 x delta, base moved", np.array(re.qpos["base"][:])-b1)
for i in range(5): env.step(np.zeros(11))
print("  5 zero steps later", np.array(re.qpos["base"][:])-b1)
# arm joint delta
a1=np.array(re.qpos["arm"][:]); a=np.zeros(11); a[3]=0.1
env.step(a); print("after +0.1 j1 delta, arm moved", np.round(np.array(re.qpos["arm"][:])-a1,4))
for i in range(3): env.step(np.zeros(11)); print("  zero:", np.round(np.array(re.qpos["arm"][:])-a1,4))
# gripper
for g in [1.0, 0.0]:
    a=np.zeros(11); a[10]=g
    for i in range(10): env.step(a)
    print("gripper cmd",g,"ctrl",re.ctrl["gripper"][:], "obs pos_gripper", float(oc._current_state.get(rob,"pos_gripper")))
# set_state exactness
env.set_state(s0); s0b=env.get_state()
print("set_state roundtrip maxdiff", np.abs(s0b-s0).max())
a=np.zeros(11); a[0]=0.05
env.set_state(s0); o1,*_=env.step(a); env.set_state(s0); o2,*_=env.step(a)
print("determinism after set_state maxdiff", np.abs(o1-o2).max())
