import numpy as np, time
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from kin import Kin, down_rot
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[4], eval_counts=[4])
s,info=env.reset(seed=0, options={'object_count': 4})
K=Kin()
def rq(s):
    r=s.get_object_from_name('robot')
    b=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
    a=np.array([s.get(r,f'pos_arm_joint{i}') for i in range(1,8)])
    return b,a,s.get(r,'pos_gripper')
import sys
gain=float(sys.argv[1])
b,a,g=rq(s)
base_t=np.array([0.6,0.0,np.pi])
t=time.time()
for i in range(8):
    b,a,g=rq(s)
    db=base_t-b; db[2]=(db[2]+np.pi)%(2*np.pi)-np.pi
    act=np.zeros(11); act[:3]=np.clip(db,-0.15,0.15)
    s,*_=env.step(act)
b,a,g=rq(s); print("base",b)
targ,err=K.ik(base_t,a,np.array([0.0,0.0,0.45]),down_rot(0.0),iters=300); print("ik err",err,targ)
for i in range(25):
    b,a,g=rq(s)
    act=np.zeros(11); act[3:10]=gain*(targ-a)
    db=base_t-b; db[2]=(db[2]+np.pi)%(2*np.pi)-np.pi; act[:3]=db
    s,*_=env.step(act)
    p,R=K.fk(b,a)
    print(i, np.abs(targ-a).max().round(4), p.round(3))
print(time.time()-t)
