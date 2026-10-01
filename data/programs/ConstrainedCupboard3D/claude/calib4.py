import sys, numpy as np, kin
from env_client import make_env
MX,MY,MZ,TOOL=0.0,0.0,0.4,0.15
GRIP=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
LIM=np.array([[-1e6,-2.41,-1e6,-2.66,-1e6,-2.23,-1e6],[1e6,2.41,1e6,2.66,1e6,2.23,1e6]])
def rq(o):
    r=o.get_object_from_name('robot'); f=o.type_features[r.type]; d=o.data[r]
    return np.array([d[f.index('pos_arm_joint%d'%(i+1))] for i in range(7)]), np.array([d[0],d[1],d[2]])
def ik_multi(T,q0,tool=TOOL):
    best=None
    for k in range(20):
        s=q0 if k==0 else np.clip(q0+np.random.uniform(-2,2,7),LIM[0],LIM[1])
        qd,ep,er=kin.ik(T,s,tool_z=tool,q_lim=LIM)
        if ep>0.004 or er>0.05: continue
        sc=np.max(np.abs(qd-q0))
        if best is None or sc<best[0]: best=(sc,qd,ep,er)
    return (None,9,9) if best is None else (best[1],best[2],best[3])
np.random.seed(0)
env=make_env(); o,info=env.reset(seed=0)
q,b=rq(o)
R=np.array([[1,0,0],[0,-1,0],[0,0,-1.0]])
# reach out in front, away from rods: px=0.55, py=-0.0 ... check rods far
for zw in [0.30,0.20,0.10,0.06,0.04,0.02,0.00,-0.02,-0.04,-0.06]:
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=[0.55,0.0,zw-MZ]
    qd,ep,er=ik_multi(T,q)
    if qd is None: print(zw,"IKfail"); continue
    for i in range(150):
        q,b=rq(o); a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qd-q,-0.1,0.1); a[10]=GRIP
        o,r,t,tr,inf=env.step(a)
        if np.max(np.abs(qd-q))<0.003: break
    q,b=rq(o)
    print("z_pred_world=%.3f steps=%d fk_err=%.4f jerr=%.4f"%(zw,i,np.linalg.norm(kin.arm_fk(q,TOOL)[:3,3]-T[:3,3]),np.max(np.abs(qd-q))))
env.close()
