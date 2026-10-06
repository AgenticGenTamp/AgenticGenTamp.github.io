import numpy as np, mujoco
from approach import Kin, HOME
k=Kin(); m=k.m; d=k.d
q=HOME.copy(); k.fk(np.zeros(3), q); mujoco.mj_forward(m,d)
sp=d.site_xpos[k.site]
for g in range(m.ngeom):
    bn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g])
    if bn and ('pad' in bn or 'follower' in bn or 'driver' in bn or 'coupler' in bn) and m.geom_contype[g]:
        # approximate vertical extent: aabb
        print(bn, m.geom_type[g], np.round(m.geom_size[g],3), "rel", np.round(d.geom_xpos[g]-sp,3), "rbound", round(m.geom_rbound[g],3))
