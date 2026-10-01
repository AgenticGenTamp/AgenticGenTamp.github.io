from env_client import make_env
from helpers import *
import numpy as np
LOG=[]
def step(env,a):
    r=env.step(a); LOG.append((r[1],r[2],r[3],r[4]))
    if r[2] and not any(l[1] for l in LOG[:-1]):
        print('TERM at',len(LOG)-1,'cubes',{k:v[:3].round(3) for k,v in cubes(r[0]).items()},'rob',rstate(r[0]).round(2))
    return r
def drive2(env,obs,target,grip,steps=200,tol=0.01):
    target=np.array(target,float)
    for t in range(steps):
        s=rstate(obs); err=target-s[:10]; err[2]=wrap(err[2]); err[3:]=wrap(err[3:])
        if np.max(np.abs(err))<tol: return obs,True
        a=np.zeros(11,np.float32); a[:10]=np.clip(err,-0.1,0.1); a[10]=grip
        obs,*_=step(env,a)
    return obs,False
def goto_cube(env,obs,pw,grip=1.0,yg=0.0):
    """move arm so grasped cube center at world pw (base fixed)"""
    s=rstate(obs); pb=world_to_base(pw,s)
    qd,e=arm_q_for(pb,yg,s[3:10])
    obs,ok=drive2(env,obs,np.concatenate([s[:3],qd]),grip,steps=250,tol=0.005)
    s2=rstate(obs); jerr=np.max(np.abs(wrap(qd-s2[3:10])))
    return obs,ok,e,jerr
def carry_base(env,obs,bx,by,byaw=0.0,grip=1.0):
    s=rstate(obs)
    return drive2(env,obs,np.concatenate([[bx,by,byaw],s[3:10]]),grip,steps=300)
def release(env,obs,n=15):
    for _ in range(n):
        a=np.zeros(11,np.float32); obs,*_=step(env,a)
    return obs
def wait(env,obs,n=30,grip=0.0):
    for _ in range(n):
        a=np.zeros(11,np.float32); a[10]=grip; obs,*_=step(env,a)
    return obs
def goto_cube(env,obs,pw,grip=1.0,yg=0.0,ds=0.02):
    s=rstate(obs); base=s[:3]
    q=s[3:10].copy()
    T=fk_arm(q); p0=np.array([T[0,3]+MX,T[1,3],T[2,3]+ZOFF])
    pb=world_to_base(pw,s)
    N=max(1,int(np.ceil(np.linalg.norm(pb-p0)/ds)))
    qd=q
    for i in range(1,N+1):
        p=p0+(pb-p0)*i/N
        qd,e=arm_q_for(p,yg,qd)
        for k in range(30):
            s=rstate(obs); err=wrap(qd-s[3:10])
            if np.max(np.abs(err))<0.01: break
            a=np.zeros(11,np.float32); a[:3]=np.clip(base-s[:3],-.1,.1); a[3:10]=np.clip(err,-0.1,0.1); a[10]=grip
            obs,*_=step(env,a)
    obs,ok=drive2(env,obs,np.concatenate([base,qd]),grip,steps=60,tol=0.005)
    s2=rstate(obs); jerr=np.max(np.abs(wrap(qd-s2[3:10])))
    return obs,ok,e,jerr
H=0.395; L=0.15
Rh=np.array([[0,0,1],[1,0,0],[0,1,0.]])
def cube_from_q(q):
    T=fk_arm(q); return T[:3,3]+np.array([MX,0,H])+L*T[:3,2]
def ikc(pb,R,q0):
    z=R[:,2]; f=pb-np.array([MX,0,H])-L*z
    q,e1,e2=ik(f,R,q0); return q,e1
def goto_h(env,obs,pw,grip=1.0,R=Rh,ds=0.02):
    s=rstate(obs); base=s[:3]; q=s[3:10].copy()
    p0=cube_from_q(q); pb=world_to_base(pw,s)
    N=max(1,int(np.ceil(np.linalg.norm(pb-p0)/ds))); qd=q
    for i in range(1,N+1):
        qd,e=ikc(p0+(pb-p0)*i/N,R,qd)
        for k in range(30):
            s=rstate(obs); err=wrap(qd-s[3:10])
            if np.max(np.abs(err))<0.01: break
            a=np.zeros(11,np.float32); a[:3]=np.clip(base-s[:3],-.1,.1); a[3:10]=np.clip(err,-0.1,0.1); a[10]=grip
            obs,*_=step(env,a)
    obs,ok=drive2(env,obs,np.concatenate([base,qd]),grip,steps=60,tol=0.005)
    s2=rstate(obs); jerr=np.max(np.abs(wrap(qd-s2[3:10])))
    return obs,ok,e,jerr
_orig=None
