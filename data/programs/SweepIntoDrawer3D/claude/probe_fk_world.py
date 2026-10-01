import numpy as np
from fk import fk, rotz
from ik import ik
RDOWN_ARM=np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)  # tool-down in arm frame
MOUNT=np.array([0.1205,0.0,0.2598])   # arm base in robot-base frame (z = fingertip-equivalent)
YAW_M=0.0
def world_from_arm(base, p_arm):
    yaw=base[2]+YAW_M
    R=rotz(base[2])
    return np.array([base[0],base[1],0.0])+R@(MOUNT+rotz(YAW_M)@np.asarray(p_arm,float))
def arm_from_world(base, p_w):
    R=rotz(base[2])
    v=R.T@(np.asarray(p_w,float)-np.array([base[0],base[1],0.0]))-MOUNT
    return rotz(YAW_M).T@v
def ee_world(obs):
    obs=np.asarray(obs,float)
    return world_from_arm(obs[125:128], fk(obs[128:135])[:3,3])
def move_world(env, obs, p_w, steps=200, grip=None, tol=0.008, gain=3.0):
    """servo EE (tool-down) to world point p_w"""
    obs=np.asarray(obs,float)
    pa=arm_from_world(obs[125:128], p_w)
    qd,err=ik(pa, RDOWN_ARM, obs[128:135])
    g=obs[135] if grip is None else grip
    prev=None; stall=0; used=0
    for i in range(steps):
        a=np.zeros(11); a[3:10]=np.clip((qd-obs[128:135])*gain,-0.1,0.1); a[10]=g
        obs,r,t,tr,_=env.step(a); obs=np.asarray(obs,float); used+=1
        e=np.abs(qd-obs[128:135]).max()
        if e<tol: break
        if prev is not None and prev-e<0.0015: stall+=1
        else: stall=0
        prev=e
        if stall>=8: break
    return obs, np.abs(qd-obs[128:135]).max(), used, err

from ik import jacobian, LIM_LO, LIM_HI
def servo_world(env, obs, p_w, steps=120, grip=None, tol=0.003, k=2.5, rot=True):
    """closed-loop resolved-rate servo of EE (tool-down) to world point p_w."""
    obs=np.asarray(obs,float); used=0; last=None
    for i in range(steps):
        q=obs[128:135].copy()
        pa=arm_from_world(obs[125:128], p_w)
        J,M=jacobian(q); ep=pa-M[:3,3]
        Rerr=RDOWN_ARM@M[:3,:3].T
        ang=np.arccos(np.clip((np.trace(Rerr)-1)/2,-1,1))
        if ang<1e-8: er=np.zeros(3)
        else:
            ax=np.array([Rerr[2,1]-Rerr[1,2],Rerr[0,2]-Rerr[2,0],Rerr[1,0]-Rerr[0,1]])/(2*np.sin(ang))
            er=ax*ang
        e=np.concatenate([ep, er if rot else np.zeros(3)])
        last=np.linalg.norm(ep)
        if last<tol and (not rot or ang<0.05): break
        lam=0.08
        dq=J.T@np.linalg.solve(J@J.T+lam**2*np.eye(6), e)
        a=np.zeros(11); a[3:10]=np.clip(dq*k,-0.1,0.1)
        a[10]=obs[135] if grip is None else grip
        obs,_,_,_,_=env.step(a); obs=np.asarray(obs,float); used+=1
    return obs, np.linalg.norm(arm_from_world(obs[125:128],p_w)-jacobian(obs[128:135])[1][:3,3]), used

def servo2(env, obs, p_w, steps=150, grip=None, tol=0.004, k=3.0, floor=0.02, rot=True):
    """resolved-rate + minimum-command floor to beat actuator deadband."""
    obs=np.asarray(obs,float); used=0
    for i in range(steps):
        q=obs[128:135].copy()
        pa=arm_from_world(obs[125:128], p_w)
        J,M=jacobian(q); ep=pa-M[:3,3]
        Rerr=RDOWN_ARM@M[:3,:3].T
        ang=np.arccos(np.clip((np.trace(Rerr)-1)/2,-1,1))
        if ang<1e-8: er=np.zeros(3)
        else:
            ax=np.array([Rerr[2,1]-Rerr[1,2],Rerr[0,2]-Rerr[2,0],Rerr[1,0]-Rerr[0,1]])/(2*np.sin(ang))
            er=ax*ang
        if np.linalg.norm(ep)<tol and (not rot or ang<0.05): break
        e=np.concatenate([ep, er if rot else np.zeros(3)])
        lam=0.08
        dq=J.T@np.linalg.solve(J@J.T+lam**2*np.eye(6), e)
        cmd=dq*k
        n=np.abs(cmd).max()
        if 0<n<floor: cmd=cmd/n*floor
        a=np.zeros(11); a[3:10]=np.clip(cmd,-0.1,0.1)
        a[10]=obs[135] if grip is None else grip
        obs,_,_,_,_=env.step(a); obs=np.asarray(obs,float); used+=1
    pa=arm_from_world(obs[125:128],p_w)
    return obs, float(np.linalg.norm(pa-jacobian(obs[128:135])[1][:3,3])), used
