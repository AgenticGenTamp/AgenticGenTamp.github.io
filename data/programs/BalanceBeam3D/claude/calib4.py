import numpy as np, sys
import kinova
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
MOUNT=np.array([0.0,0.0,0.45]); TOOL=0.19
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
def w2a(p,base):
    yaw=base[2]; c,s=np.cos(yaw),np.sin(yaw)
    R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return R.T@(p-np.array([base[0],base[1],0.0]))-MOUNT
class R:
    def __init__(s,seed=0):
        s.env=make_env(); s.obs,_=s.env.reset(seed=seed)
    def goto(s,pt,n,grip=0.0,R_=Rdown):
        for i in range(n):
            dq=kinova.ik_step(s.obs[19:26], w2a(pt,s.obs[16:19]), R_, tool_offset=TOOL)
            a=np.zeros(11); a[3:10]=np.clip(dq,-0.1,0.1); a[10]=grip
            s.obs,r,te,tr,_=s.env.step(a.astype(np.float32))
        b=s.obs[16:19]; yaw=b[2]; c,sn=np.cos(yaw),np.sin(yaw)
        Rm=np.array([[c,-sn,0],[sn,c,0],[0,0,1]])
        return np.array([b[0],b[1],0.0])+Rm@(MOUNT+kinova.fk(s.obs[19:26],TOOL)[:3,3])
    def hold(s,n,grip):
        for i in range(n):
            a=np.zeros(11); a[10]=grip
            s.obs,r,te,tr,_=s.env.step(a.astype(np.float32))

gz = float(sys.argv[1]) if len(sys.argv)>1 else 0.012
gv = float(sys.argv[2]) if len(sys.argv)>2 else 1.0
r=R(0)
blk=r.obs[54:57].copy()
print("block",blk)
print("above:",r.goto(np.array([blk[0],blk[1],0.20]),150,grip=0.0))
print("down :",r.goto(np.array([blk[0],blk[1],gz]),80,grip=0.0), "blk",r.obs[54:57],"grip",r.obs[26])
r.hold(40,gv); print("closed grip obs",r.obs[26],"blk",r.obs[54:57])
print("lift :",r.goto(np.array([blk[0],blk[1],0.25]),120,grip=gv))
print("blk after lift",r.obs[54:57])
r.env.close()
