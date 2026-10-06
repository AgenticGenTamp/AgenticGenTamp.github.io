import numpy as np, mujoco
from run_test import env
s,info=env.reset(seed=21,options={'object_count':10})
oc=env._current_backend._object_centric_env; m=oc._robot_env.sim.model.mj_model; d=oc._robot_env.sim.data.mj_data
mujoco.mj_forward(m,d)
lo=np.array([9.,9,9]); hi=-lo
for g in range(m.ngeom):
    n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g); b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g])
    nm=str(n)+'|'+str(b)
    if 'island' in nm and m.geom_contype[g]|m.geom_conaffinity[g]:
        # aabb
        c=d.geom_xpos[g]; R=d.geom_xmat[g].reshape(3,3); sz=m.geom_size[g]
        if m.geom_type[g]==6: e=np.abs(R)@sz
        else: e=np.array([sz[0]]*3)
        if c[0]+e[0]>0.85: print(nm, (c-e).round(3), (c+e).round(3))
