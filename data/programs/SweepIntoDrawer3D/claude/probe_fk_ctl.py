import numpy as np
from fk import fk, rotz
from ik import jacobian
RDOWN=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
MOUNT=np.array([0.1205,0.0,0.2598]); YAW_M=0.0
def arm_from_world(base,p_w):
    return rotz(YAW_M).T@(rotz(base[2]).T@(np.asarray(p_w,float)-np.array([base[0],base[1],0.0]))-MOUNT)
def world_from_arm(base,p_a):
    return np.array([base[0],base[1],0.0])+rotz(base[2])@(MOUNT+rotz(YAW_M)@np.asarray(p_a,float))
def ee_world(obs):
    obs=np.asarray(obs,float); return world_from_arm(obs[125:128],fk(obs[128:135])[:3,3])
def servo(env,obs,p_w,steps=80,grip=None,tol=0.002,k=1.2,wr=0.3,R=RDOWN):
    obs=np.asarray(obs,float); used=0
    for i in range(steps):
        pa=arm_from_world(obs[125:128],p_w); J,M=jacobian(obs[128:135]); ep=pa-M[:3,3]
        Rerr=R@M[:3,:3].T; ang=np.arccos(np.clip((np.trace(Rerr)-1)/2,-1,1)); ax=np.zeros(3)
        if ang>1e-8: ax=np.array([Rerr[2,1]-Rerr[1,2],Rerr[0,2]-Rerr[2,0],Rerr[1,0]-Rerr[0,1]])/(2*np.sin(ang))*ang
        if np.linalg.norm(ep)<tol and ang<0.03: break
        dq=J.T@np.linalg.solve(J@J.T+0.0064*np.eye(6),np.concatenate([ep,wr*ax]))
        a=np.zeros(11); a[3:10]=np.clip(dq*k,-0.1,0.1); a[10]=obs[135] if grip is None else grip
        obs,_,_,_,_=env.step(a); obs=np.asarray(obs,float); used+=1
    pa=arm_from_world(obs[125:128],p_w)
    return obs,float(np.linalg.norm(pa-jacobian(obs[128:135])[1][:3,3])),used
def grip_hold(env,obs,g,n=12):
    for _ in range(n):
        a=np.zeros(11); a[10]=g; obs,_,_,_,_=env.step(a)
    return np.asarray(obs,float)
