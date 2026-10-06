import numpy as np, mujoco
from kin import make_model
env, model = make_model()
obs,_=env.reset(seed=0)
ys=[obs[127]]
for t in range(10):
    a=np.zeros(11); a[2]=0.1
    obs,*_=env.step(a); ys.append(obs[127])
print("yaw steps", np.round(np.diff(ys),3))
# base geoms
for g in range(model.ngeom):
    b = model.geom_bodyid[g]
    bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
    if bn and ('base' in bn.lower()) and model.geom_contype[g]+model.geom_conaffinity[g]>0:
        print(bn, model.geom_type[g], np.round(model.geom_size[g],3), np.round(model.geom_pos[g],3))
