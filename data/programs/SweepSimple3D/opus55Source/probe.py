import numpy as np, mujoco
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SweepSimple3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube_", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[1,5,10,50], eval_counts=[1,5,10,50])
s,_=env.reset(seed=0, options={'object_count':5})
oc = env._current_backend._object_centric_env
def find(o, depth=0, path='oc'):
    if depth>4: return
    for k in dir(o):
        if k.startswith('__'): continue
        try: v=getattr(o,k)
        except Exception: continue
        if isinstance(v, mujoco.MjModel): print('MODEL', path+'.'+k); 
        elif isinstance(v, mujoco.MjData): print('DATA', path+'.'+k)
        elif hasattr(v,'__dict__') and not isinstance(v,(type,)) and 'kinder' in type(v).__module__:
            find(v, depth+1, path+'.'+k)
find(oc)
m = oc._robot_env.sim.model.mj_model; d = oc._robot_env.sim.data.mj_data
mujoco.mj_forward(m,d)
print('timestep', m.opt.timestep)
for g in range(m.ngeom):
    b = m.geom_bodyid[g]; bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    if m.geom_contype[g]==0 and m.geom_conaffinity[g]==0: continue
    if bn and ('base' in bn or 'kitchen' in bn or 'island' in bn or bn=='world'):
        # aabb in world
        c = d.geom_xpos[g]; R=d.geom_xmat[g].reshape(3,3)
        aabb = m.geom_aabb[g]  # center(3), half(3) local
        cen = c + R@aabb[:3]; half = np.abs(R)@aabb[3:]
        print(bn, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g), 'type',m.geom_type[g], 'min',np.round(cen-half,3),'max',np.round(cen+half,3))
for b in range(m.nbody):
    print('body', mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b), np.round(d.xpos[b],3)) if b<15 else None
