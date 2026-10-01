import numpy as np
from env_client import make_env
import fk, ik
from servo import *
env = make_env(); obs,info=env.reset(seed=0)
q=robot_q(obs)
# close gripper in home pose
a=np.zeros(11,dtype=np.float32); a[10]=-1.0
obs,r,t,tr,i=env.step(a); print("close@home", rinfo(obs))
a[10]=1.0
obs,r,t,tr,i=env.step(a); print("open@home", rinfo(obs))
# now descend and record joints
b=robot_base(obs); Rd=down_R(0.0)
free=np.array([0.30,-0.25])
def go(p_des,Rd,obs,q,nmax=60,tol=1e-3):
    for i in range(nmax):
        dq,ep,ew = ik.ik_step(q,p_des,Rd,tool_z=0.0,damp=0.03,max_delta=0.15)
        a=np.zeros(11,dtype=np.float32); a[3:10]=dq
        obs,rew,t,tr,inf = env.step(a)
        qn = robot_q(obs)
        blocked = np.max(np.abs(qn-(q+dq)))>1e-5
        if blocked:
            return obs,qn,ep,ew,True,q+dq
        q=qn
        if ep<tol and ew<5e-3: break
    return obs,q,ep,ew,False,None
p=np.array([free[0]-b[0],free[1]-b[1],0.30])
obs,q,ep,ew,bl,des=go(p,Rd,obs,q)
print("q above",np.round(q,3))
for Z in np.arange(0.29,0.10,-0.01):
    p[2]=Z
    obs,q,ep,ew,bl,des=go(p,Rd,obs,q,nmax=20)
    if bl:
        print("blocked at Z",round(Z,3),"q",np.round(q,3),"desired",np.round(des,3)); break
    print("Z",round(Z,3),"q",np.round(q,3))
