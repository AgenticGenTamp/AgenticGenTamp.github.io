"""Depth profile of the cupboard: push a closed gripper in +x at many heights."""
import sys, numpy as np, kin, ctrl
from env_client import make_env
SEED=int(sys.argv[1]); BAYY=float(sys.argv[2]); BASEX=float(sys.argv[3]) if len(sys.argv)>3 else 1.15
env=make_env(); o,info=env.reset(seed=SEED)
rng=np.random.default_rng(0); last=[None]
ROLLS=[0,np.pi/2,-np.pi/2,np.pi,np.pi/4,-np.pi/4,3*np.pi/4,-3*np.pi/4]
def step(a):
    global o
    o,r,t,tr,inf=env.step(np.asarray(a,dtype=np.float32)); last[0]=r
def move_base(goal,g,n=40):
    for i in range(n):
        q,b=ctrl.read_robot(o); a=np.zeros(11); a[:3]=np.clip((goal-b)/0.87,-0.1,0.1); a[10]=g
        step(a)
        if np.linalg.norm(goal-b)<0.004: break
def track(qd,g,n):
    for i in range(n):
        q,b=ctrl.read_robot(o); a=np.zeros(11); a[3:10]=np.clip((qd-q)/0.25,-0.1,0.1); a[10]=g
        step(a)
        if np.max(np.abs(qd-q))<0.004: break
def exec_traj(traj,g):
    idx=0; n=0
    while idx<len(traj) and n<len(traj)*2+30:
        q,b=ctrl.read_robot(o); a=np.zeros(11); a[3:10]=np.clip((traj[idx]-q)/0.25,-0.1,0.1); a[10]=g
        step(a); n+=1
        if np.max(np.abs(traj[idx]-q))<0.03: idx+=1
    track(traj[-1],g,8)
R=np.column_stack([np.array([0,1,0.]),np.array([0,0,1.]),np.array([1,0,0.])])
for i in range(10): step(np.concatenate([np.zeros(10),[1.0]]))
move_base(np.array([BASEX,BAYY,0.0]),1.0)
prev=None
for z in [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05]:
    q,b=ctrl.read_robot(o)
    T1=ctrl.world_to_arm(ctrl.pose_from([1.70,BAYY,z],R),b)
    qd,Tused=ctrl.ik_multi_pose(T1,q,tries=20,rng=rng,rolls=ROLLS)
    if qd is None: print("z=%.2f IKFAIL"%z); continue
    track(qd,1.0,120)
    q,b=ctrl.read_robot(o); Ta=ctrl.ee_world(q,b)
    e0=np.linalg.norm(Ta[:3,3]-np.array([1.70,BAYY,z]))
    T2=Tused.copy(); T2[0,3]+=0.45
    tr=ctrl.cartesian_traj(q,kin.arm_fk(q,ctrl.TOOL),T2)
    if tr is None: print("z=%.2f traj fail (out_err=%.3f)"%(z,e0)); continue
    exec_traj(tr,1.0)
    q,b=ctrl.read_robot(o); Tb=ctrl.ee_world(q,b)
    print("z=%.2f out_err=%.3f reached=(%.3f,%.3f,%.3f) goal_x=%.2f"%(z,e0,Tb[0,3],Tb[1,3],Tb[2,3],1.70+0.45))
env.close()
