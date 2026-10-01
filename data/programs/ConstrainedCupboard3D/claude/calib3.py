import sys, numpy as np, kin
from env_client import make_env
MX,MY,MZ,TOOL=(float(x) for x in sys.argv[1:5])
def rq(o):
    r=o.get_object_from_name('robot'); f=o.type_features[r.type]; d=o.data[r]
    return np.array([d[f.index('pos_arm_joint%d'%(i+1))] for i in range(7)]), np.array([d[0],d[1],d[2]])
LIM=np.array([[-1e6,-2.41,-1e6,-2.66,-1e6,-2.23,-1e6],[1e6,2.41,1e6,2.66,1e6,2.23,1e6]])
def ik_multi(T,q0,tool):
    best=None
    for k in range(20):
        s=q0 if k==0 else np.clip(q0+np.random.uniform(-2,2,7),LIM[0],LIM[1])
        qd,ep,er=kin.ik(T,s,tool_z=tool,q_lim=LIM)
        if ep>0.004 or er>0.05: continue
        sc=np.max(np.abs(qd-q0))
        if best is None or sc<best[0]: best=(sc,qd,ep,er)
    if best is None: return None,9,9
    return best[1],best[2],best[3]
env=make_env(); o,info=env.reset(seed=0)
q,b=rq(o)
rods=[n for n in o.get_object_names() if n.startswith('cuboid')]
tgt=min(rods,key=lambda n:np.hypot(*(o.data[o.get_object_from_name(n)][:2]-b[:2])))
rd=o.data[o.get_object_from_name(tgt)].copy()
goal=np.array([rd[0]-0.45-MX, rd[1]-MY, 0.0])
for i in range(60):
    q,b=rq(o); a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(goal-b,-0.1,0.1); a[10]=1.0
    o,r,t,tr,inf=env.step(a)
    if np.linalg.norm(goal-b)<0.003: break
q,b=rq(o)
px,py=rd[0]-b[0]-MX, rd[1]-b[1]-MY
R=np.array([[1,0,0],[0,-1,0],[0,0,-1.0]])
np.random.seed(0)
for h in [0.35,0.25,0.18,0.12,0.08,0.05,0.03,0.015,0.0,-0.02,-0.05,-0.08]:
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=[px,py,rd[2]-MZ+h]
    qd,ep,er=ik_multi(T,q,TOOL)
    if ep>0.005: print("h",h,"IKfail",round(ep,4),round(er,3)); continue
    moved=False
    for i in range(220):
        q,b=rq(o); a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qd-q,-0.1,0.1); a[10]=1.0
        o,r,t,tr,inf=env.step(a)
        rn=o.data[o.get_object_from_name(tgt)]
        if np.linalg.norm(rn[:3]-rd[:3])>0.01: moved=True; break
        if np.max(np.abs(qd-q))<0.004: break
    q,b=rq(o)
    rn=o.data[o.get_object_from_name(tgt)]
    print("h=%.3f steps=%d err=%.4f moved=%s rod=%s"%(h,i,np.linalg.norm(kin.arm_fk(q,TOOL)[:3,3]-T[:3,3]),moved,np.round(rn[:3],4)))
    if moved: break
env.close()
