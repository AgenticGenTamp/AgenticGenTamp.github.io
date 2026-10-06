import numpy as np, mujoco
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False}, design_counts=[4], eval_counts=[4])
s,info=env.reset(seed=0, options={'object_count': 4})
oc = env._current_backend._object_centric_env
re_=oc._robot_env
m=re_.sim.model.mj_model
for attr in []:
    v=getattr(re_,attr,None)
    if isinstance(v,mujoco.MjModel): m=v
    if hasattr(v,'model') and isinstance(getattr(v,'model'),mujoco.MjModel): m=v.model; d=v.data
print(type(m))
dd=mujoco.MjData(m); mujoco.mj_forward(m,dd) if False else None
for g in range(m.ngeom):
    b=m.geom_bodyid[g]; bn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b)
    if bn and 'green' in bn:
        print(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g), m.geom_type[g], m.geom_size[g].round(3), m.geom_pos[g].round(3))
for g in range(m.ngeom):
    n=m.geom(g).name or m.body(m.geom_bodyid[g]).name
    if any(k in n for k in ['pad','cube1','table','bin_red']): print('FR',n, m.geom_friction[g], m.geom_condim[g], m.geom_priority[g])
print('opt', m.opt.cone, m.opt.impratio)
