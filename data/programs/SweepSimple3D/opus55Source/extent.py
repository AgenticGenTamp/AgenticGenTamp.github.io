from ik_tool import *
import numpy as np
from approach import arm_q
for z in [0.03,0.06,0.1]:
    q=arm_q(z); fk(q)
    mx=-1;who=None
    for g in range(m.ngeom):
        b=m.geom_bodyid[g]; n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b) or ''
        if 'robot' not in n or m.geom_contype[g]==0 and m.geom_conaffinity[g]==0: continue
        x=d.geom_xpos[g][0]+m.geom_rbound[g]
        if x>mx: mx,who=x,(n,d.geom_xpos[g].round(3))
    print(z,round(mx,3),who)
