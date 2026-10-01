import numpy as np
from fk import fk
from ctrl2 import OX, L, AZ, RFWD, RDOWN
from ik import ik
from ik2 import ik_multi
from ctrl2 import tip_to_fk, JLO, JHI

def w2l(obs, wx, wy):
    bx,by,yaw=obs[125:128]
    dx,dy=wx-bx, wy-by
    return float(np.cos(yaw)*dx+np.sin(yaw)*dy), float(-np.sin(yaw)*dx+np.cos(yaw)*dy)

def l2w(obs, lx, ly):
    bx,by,yaw=obs[125:128]
    return float(bx+np.cos(yaw)*lx-np.sin(yaw)*ly), float(by+np.sin(yaw)*lx+np.cos(yaw)*ly)

def tip_local(obs):
    M=fk(obs[128:135])
    p=M[:3,3]+M[:3,:3]@np.array([0,0,L])+np.array([OX,0,0])
    return np.array([p[0],p[1],p[2]+AZ])

def tip_world(obs):
    t=tip_local(obs)
    wx,wy=l2w(obs,t[0],t[1])
    return np.array([wx,wy,t[2]])

def moveto_w(env, obs, pw, R=RFWD, steps=150, grip=0.0, tol=0.004, stall_n=20, base=None, multi=False, log=None):
    """pw = world (x,y,z) tip target."""
    lx,ly=w2l(obs,pw[0],pw[1])
    p=tip_to_fk([lx,ly,pw[2]],R)
    if multi:
        q,_c=ik_multi(np.array(p),R,obs[128:135].copy())
        if q is None:
            q,res=ik(np.array(p),R,obs[128:135].copy())
        else: res=0.0
    else:
        q,res=ik(np.array(p),R,obs[128:135].copy())
    qc=np.clip(q,JLO,JHI); clip=float(np.abs(qc-q).max()); q=qc
    hist=[]; used=0
    for i in range(steps):
        a=np.zeros(11); a[3:10]=np.clip(q-obs[128:135],-0.1,0.1); a[10]=grip
        if base is not None: a[0:3]=base
        obs,r,t,tr,_=env.step(a); used+=1
        e=float(np.abs(q-obs[128:135]).max()); hist.append(e)
        if log is not None: log.append((used,round(e,4)))
        if e<tol: break
        if len(hist)>stall_n and hist[-stall_n]-e<0.003: break
    return obs, dict(err=float(np.abs(q-obs[128:135]).max()), used=used, ikres=float(res), clip=clip,
                     tip=np.round(tip_world(obs),4), qd=q)

def hold(env,obs,q,n,grip,base=None):
    for i in range(n):
        a=np.zeros(11); a[3:10]=np.clip(q-obs[128:135],-0.1,0.1); a[10]=grip
        if base is not None: a[0:3]=base
        obs,r,t,tr,_=env.step(a)
    return obs
