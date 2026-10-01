import numpy as np, pickle, sys
from env_client import make_env
import fk, ik
from servo import *
def go(env,obs,q,p_des,Rd,nmax=40,tol=1e-3):
    for i in range(nmax):
        dq,ep,ew = ik.ik_step(q,p_des,Rd,tool_z=0.0,damp=0.03,max_delta=0.15)
        a=np.zeros(11,dtype=np.float32); a[3:10]=dq
        obs,rew,t,tr,inf = env.step(a)
        qn = robot_q(obs)
        if np.max(np.abs(qn-(q+dq)))>1e-5: return obs,qn,ep,ew,True
        q=qn
        if ep<tol and ew<5e-3: break
    return obs,q,ep,ew,False
seed=int(sys.argv[1])
rng=np.random.default_rng(seed)
env=make_env(); obs,info=env.reset(seed=seed); q=robot_q(obs); b=robot_base(obs)
blk=opos(obs,'target_block'); Rd=down_R(0.0)
p=np.array([blk[0]-b[0]-0.12, blk[1]-b[1], blk[2]+0.185])
obs,q,ep,ew,bl=go(env,obs,q,p,Rd,nmax=60)
a=np.zeros(11,dtype=np.float32); a[10]=-1.0
obs,r,t,tr,inf=env.step(a)
ri=rinfo(obs)
print("grasp?",ri['grasp_active'],"tf",ri, flush=True)
if ri['grasp_active']<0.5: sys.exit(1)
G = grasp_tf_mat(obs)
data=[]
q=robot_q(obs)
for it in range(250):
    dq = rng.uniform(-0.15,0.15,7)
    a=np.zeros(11,dtype=np.float32); a[3:10]=dq
    obs,r,t,tr,inf=env.step(a)
    qn=robot_q(obs)
    if np.max(np.abs(qn-(q+dq)))>1e-5:
        # blocked; try opposite
        a[3:10]=-dq*0.5
        obs,r,t,tr,inf=env.step(a); qn=robot_q(obs)
    q=qn
    if obs.get_object_from_name('robot') is not None and rinfo(obs)['grasp_active']<0.5:
        print("lost grasp at",it); break
    data.append((q.copy(), opose(obs,'target_block'), robot_base(obs)))
pickle.dump({'data':data,'G':G,'base':b}, open(f'calib_{seed}.pkl','wb'))
print("collected",len(data))
