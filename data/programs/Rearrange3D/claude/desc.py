import sys
import numpy as np
from env_client import make_env
import kinova_fk as K
np.set_printoptions(precision=4, suppress=True)
MOUNT=np.array([0.,0.,0.44])
G=float(sys.argv[1]); MODE=sys.argv[2]  # 'can' or 'table'
env=make_env(); obs,_=env.reset(seed=0); obs=np.asarray(obs,float)
def step(a):
    global obs
    o,r,te,tr,i=env.step(np.asarray(a,float)); obs=np.asarray(o,float)
def w2r(p):
    bx,by,th=obs[93],obs[94],obs[95]; c,s=np.cos(-th),np.sin(-th)
    d=np.asarray(p,float)-np.array([bx,by,0.]); return np.array([c*d[0]-s*d[1],s*d[0]+c*d[1],d[2]])-MOUNT
def r2w(p):
    bx,by,th=obs[93],obs[94],obs[95]; p=np.asarray(p,float)+MOUNT; c,s=np.cos(th),np.sin(th)
    return np.array([bx+c*p[0]-s*p[1],by+s*p[0]+c*p[1],p[2]])
def ee(): return r2w(K.fk(obs[96:103])[:3,3])
Rd=np.array([[1.,0,0],[0,-1.,0],[0,0,-1.]])
def servo(tw,n,g):
    ok=None;pe=None
    for _ in range(n):
        qt,ok,inf=K.ik(w2r(tw),target_rot=Rd,q_init=obs[96:103],restarts=2,max_iters=200)
        pe=inf['pos_err']
        a=np.zeros(11); a[3:10]=np.clip((qt-obs[96:103])*4.0,-0.1,0.1); a[10]=g
        step(a)
    return ee(),ok,pe
SL=32; can=obs[SL:SL+3].copy()
xy = can[:2] if MODE=='can' else np.array([can[0], can[1]+0.16])
print("mode",MODE,"xy",xy,"can",can)
# gripper set BEFORE motion, high above
a=np.zeros(11); a[10]=G
for _ in range(10): step(a)
e,ok,pe=servo(np.array([xy[0],xy[1],0.70]),120,G)
print("above EE",e,"ikok",ok,"ikerr",round(pe,5))
blocked=None
for z in np.arange(0.69,0.44,-0.01):
    tw=np.array([xy[0],xy[1],z]); e,ok,pe=servo(tw,12,G)
    err=e[2]-z
    if err>0.022:
        blocked=(z,e[2]); print("BLOCKED cmd_z=%.3f achieved_fkz=%.4f canpos=%s"%(z,e[2],obs[SL:SL+3])); break
if blocked is None: print("never blocked, final",e,"can",obs[SL:SL+3])
print("G=%.1f MODE=%s BLOCK_FKZ=%s can_moved=%.4f"%(G,MODE,None if blocked is None else round(blocked[1],4),np.linalg.norm(obs[SL:SL+3]-can)))
env.close()
