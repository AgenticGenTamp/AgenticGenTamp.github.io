import time, numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
t=time.time(); env=KinderGeom3DEnv("kinder/BalanceBeam3D-o3-v0"); print("make",time.time()-t)
print(env.action_space, env.observation_space.shape)
ke=env._kinder_env
print(type(ke), getattr(ke,'spec',None))
oc=ke._object_centric_env
for s in range(5):
    t=time.time(); obs,_=env.reset(seed=s); dt=time.time()-t
    st=oc._current_state
    print(f"--seed {s} reset {dt:.2f}s")
    for o in sorted(st, key=lambda o:o.name):
        f=o.type.feature_names
        if o.name.startswith("robot") or 'pos_base_x' in f:
            print(" ",o.name, {k:round(float(st.get(o,k)),3) for k in f if k.startswith('pos')})
        else:
            print(" ",o.name, {k:round(float(st.get(o,k)),4) for k in ['x','y','z','qw','qx','qy','qz']})
    if s==0:
        print("obs names:", [ (o.name, o.type.feature_names) for o in st][:1])
        print("obs len", obs.shape)
