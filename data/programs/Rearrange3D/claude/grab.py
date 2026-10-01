import sys, numpy as np
from env_client import make_env
import kinova_fk as K
np.set_printoptions(precision=4, suppress=True)
MOUNT=np.array([0.,0.,0.44]); SL=32
G=float(sys.argv[1])
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
def Rd(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])@np.array([[1.,0,0],[0,-1.,0],[0,0,-1.]])
def servo(tw,n,g,yaw=0.0):
    for _ in range(n):
        qt,ok,inf=K.ik(w2r(tw),target_rot=Rd(yaw),q_init=obs[96:103],restarts=1,max_iters=120)
        a=np.zeros(11); a[3:10]=np.clip((qt-obs[96:103])*4.0,-0.1,0.1); a[10]=g
        step(a)
    return ee()
can0=obs[SL:SL+3].copy()
OPEN=1.0-G
servo(can0+np.array([0,0,0.20]),90,OPEN)          # above can, gripper "open"
e=servo(can0+np.array([0,0,0.02]),80,OPEN)        # descend onto can
c1=obs[SL:SL+3].copy()
a=np.zeros(11); a[10]=G
for _ in range(10): step(a)                        # close
c2=obs[SL:SL+3].copy()
# drive base -x 0.87m while holding
a=np.zeros(11); a[0]=-0.1; a[10]=G
b0=obs[93]
for _ in range(10): step(a)
c3=obs[SL:SL+3].copy(); db=obs[93]-b0
print(f"G={G} ee_at_can={e} err={np.linalg.norm(e-can0):.3f} can_after_descend={c1} after_close={c2}")
print(f"   base_dx={db:.3f} can_dx={c3[0]-c2[0]:.3f} can_after_basemove={c3} FOLLOWS={'YES' if abs((c3[0]-c2[0])-db)<0.2 else 'no'}")
env.close()
