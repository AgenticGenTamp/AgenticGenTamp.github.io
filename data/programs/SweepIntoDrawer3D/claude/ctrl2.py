import numpy as np
from ik import ik
RDOWN=np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)
RFWD=np.array([[0,0,1],[0,1,0],[-1,0,0]],dtype=float)
JLO=np.array([-1e9,-2.41,-1e9,-2.66,-1e9,-2.23,-1e9])
JHI=np.array([ 1e9, 2.41, 1e9, 2.66, 1e9, 2.23, 1e9])
OX=0.10; L=0.16; AZ=0.43   # calibration params (tip offset in base frame, tool len, arm base height)

def tip_to_fk(p_tip, R):
    """p_tip: desired tip pos in (local x, local y, world z). returns fk target pos"""
    p=np.array(p_tip,dtype=float).copy()
    p[2]=p[2]-AZ
    p=p - R@np.array([0,0,L]) - np.array([OX,0,0])
    return p

def moveto(env, obs, p_tip, R=RDOWN, steps=200, grip=None, tol=0.004, base=None, stall_n=15):
    p=tip_to_fk(p_tip,R)
    qd,res=ik(np.array(p),R,obs[128:135].copy())
    qdc=np.clip(qd,JLO,JHI)
    clipped=np.abs(qdc-qd).max()
    qd=qdc
    g = obs[135] if grip is None else grip
    used=0; hist=[]
    for i in range(steps):
        a=np.zeros(11); a[3:10]=np.clip(qd-obs[128:135],-0.1,0.1); a[10]=g
        if base is not None: a[0:3]=base
        obs,r,t,tr,_=env.step(a); used+=1
        err=np.abs(qd-obs[128:135]).max(); hist.append(err)
        if err<tol: break
        if len(hist)>stall_n and hist[-stall_n]-err < 0.004: break
    return obs, dict(err=float(np.abs(qd-obs[128:135]).max()), used=used, ikres=float(res), clip=float(clipped))

def move_base(env, obs, target, steps=40, grip=None, qhold=None):
    g = obs[135] if grip is None else grip
    for i in range(steps):
        a=np.zeros(11)
        cur=obs[125:128]
        d=np.array(target,dtype=float)-cur
        d[2]=(d[2]+np.pi)%(2*np.pi)-np.pi
        a[0]=np.clip(d[0]/0.8704,-0.1,0.1); a[1]=np.clip(d[1]/0.8704,-0.1,0.1); a[2]=np.clip(d[2]/0.994,-0.1,0.1)
        if qhold is not None: a[3:10]=np.clip(qhold-obs[128:135],-0.1,0.1)
        a[10]=g
        obs,r,t,tr,_=env.step(a)
        if np.abs(d[:2]).max()<0.003 and abs(d[2])<0.01: break
    return obs
