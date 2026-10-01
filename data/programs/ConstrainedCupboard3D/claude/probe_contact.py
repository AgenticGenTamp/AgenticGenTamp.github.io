"""Poke the closed gripper horizontally into the cupboard at many heights."""
import sys, numpy as np, kin, ctrl
from env_client import make_env
SEED=int(sys.argv[1]); BAYY=float(sys.argv[2])
env=make_env(); o,info=env.reset(seed=SEED)
rng=np.random.default_rng(0)
last=[None]
def step(a):
    global o
    o,r,t,tr,inf=env.step(np.asarray(a,dtype=np.float32)); last[0]=r; return r
def move_base(goal,g,n=40):
    for i in range(n):
        q,b=ctrl.read_robot(o); a=np.zeros(11); a[:3]=np.clip((goal-b)/0.87,-0.1,0.1); a[10]=g
        step(a)
        if np.linalg.norm(goal-b)<0.004: break
def exec_traj(traj,g,extra=6):
    idx=0; n=0
    while idx<len(traj) and n<len(traj)*3+40:
        q,b=ctrl.read_robot(o); a=np.zeros(11); a[3:10]=np.clip((traj[idx]-q)/0.25,-0.1,0.1); a[10]=g
        step(a); n+=1
        if np.max(np.abs(traj[idx]-q))<0.03: idx+=1
    for i in range(extra):
        q,b=ctrl.read_robot(o); a=np.zeros(11); a[3:10]=np.clip((traj[-1]-q)/0.25,-0.1,0.1); a[10]=g
        step(a)
    return ctrl.read_robot(o)
def goto_pose(T_world,g,tries=16):
    q,b=ctrl.read_robot(o)
    T0=kin.arm_fk(q,ctrl.TOOL); T1=ctrl.world_to_arm(T_world,b)
    qd=ctrl.ik_multi(T1,q,tries=tries,rng=rng)
    if qd is None: return None,False
    tr=ctrl.cartesian_traj(q,T0,T1)
    if tr is None:
        tr=[q+(qd-q)*(i/40.) for i in range(1,41)]
    q,b=exec_traj(tr,g)
    Tach=ctrl.ee_world(q,b)
    return Tach, np.linalg.norm(Tach[:3,3]-T_world[:3,3])<0.02
# gripper closed, pointing +x:  z_tool = +x_world
R=np.column_stack([np.array([0,1,0.]),np.array([0,0,1.]),np.array([1,0,0.])])
for i in range(10): step(np.concatenate([np.zeros(10),[1.0]]))
move_base(np.array([1.15,BAYY,0.0]),1.0)
print("base",np.round(ctrl.read_robot(o)[1],3))
for z in [0.85,0.75,0.65,0.55,0.45,0.35,0.25,0.15,0.08]:
    Tach,ok=goto_pose(ctrl.pose_from([1.72,BAYY,z],R),1.0)
    if Tach is None: print(z,"ik fail out"); continue
    ok0=np.linalg.norm(Tach[:3,3]-np.array([1.72,BAYY,z]))<0.02
    Tach2,ok2=goto_pose(ctrl.pose_from([2.02,BAYY,z],R),1.0)
    if Tach2 is None: print(z,"ik fail in"); continue
    print("z=%.2f out_ok=%s reached_x=%.3f err=%.3f rew=%s"%(z,ok0,Tach2[0,3],np.linalg.norm(Tach2[:3,3]-np.array([2.02,BAYY,z])),last[0]))
env.close()
