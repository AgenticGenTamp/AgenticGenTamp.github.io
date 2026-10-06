import mujoco, numpy as np
from approach import Kin
k=Kin(); m=k.m
for i in range(m.nu):
    n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i)
    if 'grip' in (n or '') or 'finger' in (n or '') or i>=m.nu-2: print('act',i,n,m.actuator_gainprm[i][:3],m.actuator_biasprm[i][:3],m.actuator_forcerange[i],m.actuator_ctrlrange[i])
for g in range(m.ngeom):
    b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g])
    if 'pad' in b: print('geom',b,m.geom_type[g],m.geom_size[g],m.geom_friction[g],m.geom_condim[g])
b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'bin_yellow_0'); 
for g in range(m.ngeom):
    if m.geom_bodyid[g]==b: print('bin geom fr',m.geom_friction[g],m.geom_condim[g]); break
