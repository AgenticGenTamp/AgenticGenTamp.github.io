import numpy as np, kin, sys
from lib import *
from env_client import make_env
env=make_env(); o,info=env.reset(seed=0)
q,b=rq(o)
rods=sorted([n for n in o.get_object_names() if n.startswith('cuboid')])
tgt=min(rods,key=lambda n:np.hypot(*(o.data[o.get_object_from_name(n)][:2]-b[:2])))
rd=o.data[o.get_object_from_name(tgt)].copy()
yaw=np.arctan2(2*rd[3]*rd[6],1-2*rd[6]**2); u=np.array([-np.sin(yaw),np.cos(yaw),0.0])
gr=[0.0]; last=[None]
def step(a):
    global o
    o,r,t,tr,inf=env.step(a); last[0]=(r,t); return r
def move_base(goal,n=40):
    for i in range(n):
        q,b=rq(o); a=np.zeros(11,dtype=np.float32); a[:3]=np.clip((goal-b)/0.87,-0.1,0.1); a[10]=gr[0]
        step(a)
        if np.linalg.norm(goal-b)<0.004: break
def arm_to(R,pos,g,n=80):
    gr[0]=g
    q,b=rq(o); T=world_to_arm(pos,R,b)
    qd=ik_multi(T,q)
    if qd is None: return False
    for i in range(n):
        q,b=rq(o); a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip((qd-q)/0.25,-0.1,0.1); a[10]=g
        step(a)
        if np.max(np.abs(qd-q))<0.002: break
    return True
move_base(np.array([rd[0]-0.5-MOUNT[0],rd[1],0.0]))
z=np.array([0,0,-1.0]); x=np.array([u[1],-u[0],0.0]); y=np.cross(z,x); R=np.column_stack([x,y,z])
print("hover",arm_to(R,[rd[0],rd[1],rd[2]+0.12],0.0))
print("desc",arm_to(R,[rd[0],rd[1],rd[2]],0.0,60))
for i in range(8): step(np.concatenate([np.zeros(10),[1.0]]).astype(np.float32))
gr[0]=1.0
print("lift",arm_to(R,[rd[0],rd[1],0.4],1.0))
rn=o.data[o.get_object_from_name(tgt)]; print("rod after lift",np.round(rn[:3],3),"rew",last[0])
# sample positions
for pos in [[rd[0],rd[1],0.4],[rd[0]+0.3,rd[1],0.4],[rd[0],rd[1]+0.3,0.4],[rd[0],rd[1],0.7],[rd[0]-0.3,rd[1],0.5],[rd[0],rd[1]-0.3,0.6]]:
    ok=arm_to(R,pos,1.0)
    rn=o.data[o.get_object_from_name(tgt)]
    print("ok",ok,"rod",np.round(rn[:3],3),"rew",round(last[0][0],4),"term",last[0][1])
env.close()
