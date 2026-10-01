import numpy as np, sys
from env_client import make_env
from fk import fk
from ik import solve

def target_T(pos, yaw=0.0):
    # gripper z axis pointing down
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.0]])
    Rd = np.array([[1,0,0],[0,-1,0],[0,0,-1.0]])  # x fwd, z down
    T=np.eye(4); T[:3,:3]=Rz@Rd; T[:3,3]=pos
    return T

def run(tool_len, seed=0, zoff=0.0, verbose=True):
    env=make_env(); obs,info=env.reset(seed=seed)
    c=obs.get_object_from_name("cube0")
    cp=np.array([float(obs.get(c,f)) for f in ["pose_x","pose_y","pose_z"]])
    q=np.array([float(obs.get(obs.get_object_from_name("robot"),f)) for f in ["joint_%d"%i for i in range(1,8)]])
    # pregrasp 0.15 above
    Tpre=target_T(cp+np.array([0,0,0.15+zoff]))
    qpre,pe,oe=solve(Tpre,tool_len)
    Tg=target_T(cp+np.array([0,0,zoff]))
    qg,pe2,oe2=solve(Tg,tool_len,seeds=[qpre])
    if verbose: print("tool",tool_len,"ik errs",round(pe,4),round(oe,4),"|",round(pe2,4),round(oe2,4))
    def goto(qt,nmax=60):
        nonlocal q,obs
        for _ in range(nmax):
            d=qt-q
            if np.max(np.abs(d))<1e-3: return True
            step=np.clip(d,-0.35,0.35)
            a=np.zeros(11); a[3:10]=step
            obs,r,t,tr,i=env.step(a)
            qn=np.array([float(obs.get(obs.get_object_from_name("robot"),f)) for f in ["joint_%d"%k for k in range(1,8)]])
            if np.allclose(qn,q,atol=1e-6):
                return False  # rejected
            q=qn
        return False
    ok1=goto(qpre); ok2=goto(qg)
    a=np.zeros(11); a[10]=-1.0
    obs,r,t,tr,i=env.step(a)
    rb=obs.get_object_from_name("robot")
    ga=float(obs.get(rb,"grasp_active")); fs=float(obs.get(rb,"finger_state"))
    cz=float(obs.get(obs.get_object_from_name("cube0"),"pose_z"))
    if verbose: print("  pre_ok",ok1,"grasp_ok",ok2,"grasp_active",ga,"finger",fs,"cubez",cz,"term",t)
    env.close()
    return ga

if __name__=="__main__":
    for tl in [0.0,0.06,0.09,0.12,0.15,0.18]:
        run(tl)
