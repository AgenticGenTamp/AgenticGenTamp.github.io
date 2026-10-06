import numpy as np, mujoco
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[4], eval_counts=[4])
s,info=env.reset(seed=0, options={'object_count': 4})
oc = env._current_backend._object_centric_env
sim = oc._robot_env.sim; m=sim.model.mj_model; d=sim.data.mj_data
sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'robot_pinch_site')
for a in [0.0,0.45,1.0]:
    act=np.zeros(11); act[10]=a
    for i in range(3): s,*_=env.step(act)
    ps=d.site_xpos[sid]; R=d.site_xmat[sid].reshape(3,3)
    out=[]
    for g in range(m.ngeom):
        b=m.geom_bodyid[g]; bn=m.body(b).name
        if not any(k in bn for k in ['pad','follower','coupler','driver','spring','robot_base']) : continue
        if m.geom_contype[g]==0 and m.geom_conaffinity[g]==0: continue
        # local coords in site frame
        rel=R.T@(d.geom_xpos[g]-ps)
        out.append((bn, m.geom_type[g], rel.round(3), m.geom_size[g].round(3)))
    print("cmd",a)
    for o in out: print("  ",o)
