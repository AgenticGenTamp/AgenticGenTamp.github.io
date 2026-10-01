import sys, numpy as np, kin
from env_client import make_env
MX,MY,MZ,TOOL,AXIS=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),int(sys.argv[5])
SEED=int(sys.argv[6]) if len(sys.argv)>6 else 0
LIM=np.array([[-1e6,-2.41,-1e6,-2.66,-1e6,-2.23,-1e6],[1e6,2.41,1e6,2.66,1e6,2.23,1e6]])
def rq(o):
    r=o.get_object_from_name('robot'); f=o.type_features[r.type]; d=o.data[r]
    return np.array([d[f.index('pos_arm_joint%d'%(i+1))] for i in range(7)]), np.array([d[0],d[1],d[2]]), d[f.index('pos_gripper')]
def ik_multi(T,q0):
    best=None
    for k in range(25):
        s=q0 if k==0 else np.clip(q0+np.random.uniform(-2,2,7),LIM[0],LIM[1])
        qd,ep,er=kin.ik(T,s,tool_z=TOOL,q_lim=LIM)
        if ep>0.004 or er>0.05: continue
        sc=np.max(np.abs(qd-q0))
        if best is None or sc<best[0]: best=(sc,qd)
    return None if best is None else best[1]
np.random.seed(0)
env=make_env(); o,info=env.reset(seed=SEED)
q,b,g=rq(o)
rods=[n for n in o.get_object_names() if n.startswith('cuboid')]
tgt=min(rods,key=lambda n:np.hypot(*(o.data[o.get_object_from_name(n)][:2]-b[:2])))
rd=o.data[o.get_object_from_name(tgt)].copy()
yaw=np.arctan2(2*rd[3]*rd[6],1-2*rd[6]**2)
u=np.array([-np.sin(yaw),np.cos(yaw),0.0])
def move_base(o,goal,n=80):
    for i in range(n):
        q,b,g=rq(o); a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(goal-b,-0.1,0.1); a[10]=gr
        o,r,t,tr,inf=env.step(a)
        if np.linalg.norm(goal-b)<0.004: break
    return o
gr=0.0
o=move_base(o,np.array([rd[0]-0.5-MX,rd[1]-MY,0.0]))
q,b,g=rq(o)
z=np.array([0,0,-1.0])
x=np.array([u[1],-u[0],0.0]) if AXIS==0 else u.copy()
y=np.cross(z,x)
R=np.column_stack([x,y,z])
def arm_to(o,pos,gripv,n=150):
    global gr
    gr=gripv
    q,b,g=rq(o)
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=[pos[0]-b[0]-MX,pos[1]-b[1]-MY,pos[2]-MZ]
    qd=ik_multi(T,q)
    if qd is None: return o,False
    for i in range(n):
        q,b,g=rq(o); a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qd-q,-0.1,0.1); a[10]=gripv
        o,r,t,tr,inf=env.step(a)
        if np.max(np.abs(qd-q))<0.003: break
    return o,True
ok1=arm_to(o,[rd[0],rd[1],rd[2]+0.12],0.0); o=ok1[0]
print("hover ok",ok1[1], "rod",np.round(o.data[o.get_object_from_name(tgt)][:3],4))
o,ok=arm_to(o,[rd[0],rd[1],rd[2]],0.0,120)
print("descend ok",ok,"rod",np.round(o.data[o.get_object_from_name(tgt)][:3],4))
for i in range(40):
    a=np.zeros(11,dtype=np.float32); a[10]=1.0; o,r,t,tr,inf=env.step(a)
print("closed grip", round(rq(o)[2],3), "rod",np.round(o.data[o.get_object_from_name(tgt)][:3],4))
o,ok=arm_to(o,[rd[0],rd[1],rd[2]+0.3],1.0,150)
rn=o.data[o.get_object_from_name(tgt)]
print("RESULT MX=%.2f MY=%.2f MZ=%.2f TOOL=%.2f AXIS=%d lift_z=%.4f rod=%s"%(MX,MY,MZ,TOOL,AXIS,rn[2],np.round(rn[:3],3)))
env.close()
