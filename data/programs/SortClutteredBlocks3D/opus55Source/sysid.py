import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[4], eval_counts=[4])
from kin import Kin, down_rot
K=Kin()
s,info=env.reset(seed=0, options={'object_count': 4})
def rq(s):
    r=s.get_object_from_name('robot')
    b=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
    a=np.array([s.get(r,f'pos_arm_joint{i}') for i in range(1,8)])
    v=np.array([s.get(r,f'vel_arm_joint{i}') for i in range(1,8)])
    return b,a,v
base_t=np.array([0.6,0,np.pi])
rng=np.random.default_rng(0)
X=[];Y=[]
b,a,v=rq(s)
home,_=K.ik(base_t,np.array([0.0,-0.35,3.14,-2.55,0.0,-0.87,1.57]),np.array([0.0,0.0,0.5]),down_rot(0),iters=300)
for ep in range(6):
    # go near a random reachable config
    tgt,_=K.ik(base_t,home,np.array([rng.uniform(-0.12,0.12),rng.uniform(-0.15,0.15),rng.uniform(0.45,0.6)]),down_rot(rng.uniform(-1,1)),iters=300)
    for k in range(15):
        b,a,v=rq(s); act=np.zeros(11); db=base_t-b; db[2]=(db[2]+np.pi)%(2*np.pi)-np.pi; act[:3]=db
        act[3:10]=1.5*(tgt-a); s,*_=env.step(act)
    for k in range(25):
        b,a,v=rq(s)
        d=rng.uniform(-0.15,0.15,7)+0.5*(tgt-a)
        act=np.zeros(11); act[3:10]=d
        s,*_=env.step(act)
        b2,a2,v2=rq(s)
        X.append(np.stack([d,v],1)); Y.append(np.stack([a2-a,v2],1))
X=np.array(X); Y=np.array(Y)
for j in range(7):
    Xj=X[:,j,:]; 
    for o,name in [(0,'dq'),(1,'v1')]:
        coef,res,_,_=np.linalg.lstsq(Xj,Y[:,j,o],rcond=None)
        pred=Xj@coef; r2=1-np.var(Y[:,j,o]-pred)/np.var(Y[:,j,o])
        print(j,name,coef.round(4),'r2',round(r2,4))
