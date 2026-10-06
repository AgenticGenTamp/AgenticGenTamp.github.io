import numpy as np, time
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from kin import Kin, down_rot
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[4], eval_counts=[4])
K=Kin()
def rq(s):
    r=s.get_object_from_name('robot')
    b=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
    a=np.array([s.get(r,f'pos_arm_joint{i}') for i in range(1,8)])
    v=np.array([s.get(r,f'vel_arm_joint{i}') for i in range(1,8)])
    return b,a,v
base_t=np.array([0.6,0.0,np.pi])
s,info=env.reset(seed=0, options={'object_count': 4})
for i in range(8):
    b,a,v=rq(s); db=base_t-b; db[2]=(db[2]+np.pi)%(2*np.pi)-np.pi
    act=np.zeros(11); act[:3]=db; s,*_=env.step(act)
saved=env.get_state()
b,a,v=rq(s)
t1,_=K.ik(base_t,a,np.array([0.0,0.0,0.45]),down_rot(0.0),iters=300)
t2,_=K.ik(base_t,t1,np.array([-0.1,0.15,0.52]),down_rot(0.5),iters=300)
for g1,g2 in [(1.5,1.5),(2.0,1.5),(1.5,2.0),(1.8,1.8),(2.0,1.0),(1.0,2.0)]:
  for dd in [0.0]:
    g=np.array([g1]*4+[g2]*3)
    env.reset(seed=0, options={'object_count': 4}); env.set_state(saved); s=saved
    res=[]
    for targ in [t1,t2]:
        hit=None
        for i in range(30):
            b,a,v=rq(s)
            act=np.zeros(11); act[3:10]=g*(targ-a)-dd*v
            db=base_t-b; db[2]=(db[2]+np.pi)%(2*np.pi)-np.pi; act[:3]=db
            s,*_=env.step(act)
            b,a,v=rq(s)
            if np.abs(targ-a).max()<0.02 and hit is None: hit=i
        res.append((hit, np.abs(targ-a).max().round(3)))
    print(g1,g2,res)
