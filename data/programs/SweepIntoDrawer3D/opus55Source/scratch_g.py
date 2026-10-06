import numpy as np, mujoco
from kin import *
env,m=make_model()
re=env._kinder_env._object_centric_env._robot_env
d=re.sim.data.mj_data
sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'robot_pinch_site')
for gval in [0.0,0.5,1.0]:
    for _ in range(15): env.step(np.r_[np.zeros(10),gval])
    R=d.site_xmat[sid].reshape(3,3); P=d.site_xpos[sid]
    lo=np.full(3,9.);hi=-lo
    for g in range(m.ngeom):
        bn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ''
        if bn.startswith('robot_') and m.geom_contype[g]+m.geom_conaffinity[g]>0 and 'link' not in bn and 'base' not in bn:
            c=m.geom_aabb[g][:3]; h=m.geom_aabb[g][3:]
            Rg=R.T@d.geom_xmat[g].reshape(3,3); p=R.T@(d.geom_xpos[g]-P)+Rg@c; e=np.abs(Rg)@h
            if gval==0.5: print(bn,(p-e).round(3),(p+e).round(3))
            lo=np.minimum(lo,p-e);hi=np.maximum(hi,p+e)
    print('grip',gval,lo.round(3),hi.round(3))
