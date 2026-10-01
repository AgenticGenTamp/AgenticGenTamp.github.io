"""Calibrated Cartesian control helpers (test-side)."""
import numpy as np
import kinova_fk as K

Z_OFF = 0.267
XY_OFF = np.array([0.0,0.0])

def R_down(yaw=0.0):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])@np.array([[1,0,0],[0,-1,0],[0,0,-1.]])

def world_to_rel(p, base):
    bx,by,th=base[0],base[1],base[2]
    v=np.array([p[0]-bx,p[1]-by]); c,s=np.cos(th),np.sin(th)
    return np.array([c*v[0]+s*v[1]-XY_OFF[0], -s*v[0]+c*v[1]-XY_OFF[1], p[2]-Z_OFF])

def rel_to_world(r, base):
    bx,by,th=base[0],base[1],base[2]; c,s=np.cos(th),np.sin(th)
    x=r[0]+XY_OFF[0]; y=r[1]+XY_OFF[1]
    return np.array([bx+c*x-s*y, by+s*x+c*y, r[2]+Z_OFF])

def tcp_world(obs):
    o=np.asarray(obs); return rel_to_world(K.fk(o[96:103])[:3,3], o[93:96])

def step_to(env,obs,qt,steps,grip,kp=3.0,rec=None):
    for i in range(steps):
        a=np.zeros(11,dtype=np.float32); a[10]=grip
        q=np.asarray(obs)[96:103]
        a[3:10]=np.clip(kp*(qt-q),-0.1,0.1)
        obs,r,te,tr,inf=env.step(a)
        if rec is not None: rec.append((float(r),np.asarray(obs).copy()))
    return obs

def move_world(env,obs,p,steps=10,grip=0.0,yaw=0.0,maxjump=1.5,rec=None):
    o=np.asarray(obs); q=o[96:103].copy()
    rel=world_to_rel(np.asarray(p,float),o[93:96])
    qt,ok,inf=K.ik(rel,R_down(yaw),q_init=q)
    if not ok or inf['pos_err']>0.005: return obs,False
    if np.max(np.abs(qt-q))>maxjump: return obs,False
    return step_to(env,obs,qt,steps,grip,rec=rec),True

def move_world_line(env,obs,p0,p1,n=6,steps_per=6,grip=0.0,yaw=0.0,rec=None,maxjump=1.5):
    oks=0
    for t in np.linspace(0,1,n+1)[1:]:
        p=(1-t)*np.asarray(p0,float)+t*np.asarray(p1,float)
        obs,ok=move_world(env,obs,p,steps_per,grip,yaw,maxjump=maxjump,rec=rec)
        oks+=ok
    return obs,oks

def base_goto(env,obs,tx,ty,tth=0.0,steps=40,grip=0.0,tol=0.008,rec=None,vmax=0.1,stall_tol=0.004):
    prev=None; stall=0
    for i in range(steps):
        o=np.asarray(obs); b=o[93:96].copy()
        e=np.array([tx-b[0],ty-b[1],(tth-b[2]+np.pi)%(2*np.pi)-np.pi])
        if np.max(np.abs(e))<tol: break
        if prev is not None and np.max(np.abs(b-prev))<stall_tol:
            stall+=1
            if stall>=3: break
        else: stall=0
        prev=b
        a=np.zeros(11,dtype=np.float32); a[10]=grip
        a[0:3]=np.clip(np.array([1.2,1.2,1.5])*e,-vmax,vmax)
        obs,r,te,tr,inf=env.step(a)
        if rec is not None: rec.append((float(r),np.asarray(obs).copy()))
    return obs

def hold(env,obs,n,grip,qt=None,rec=None):
    return step_to(env,obs,qt if qt is not None else np.asarray(obs)[96:103].copy(),n,grip,rec=rec)

def servo(env,obs,p,grip=0.0,yaw=0.0,tol=0.006,max_steps=60,chunk=6,rec=None,kp=3.0):
    """Servo TCP to world point p using FK feedback with integral correction."""
    p=np.asarray(p,float); bias=np.zeros(3); used=0
    while used<max_steps:
        o=np.asarray(obs)
        cur=tcp_world(o)
        err=p-cur
        if np.linalg.norm(err)<tol: break
        bias=bias+0.8*err
        bias=np.clip(bias,-0.2,0.2)
        rel=world_to_rel(p+bias,o[93:96])
        qt,ok,inf=K.ik(rel,R_down(yaw),q_init=o[96:103])
        if not ok or inf['pos_err']>0.01:
            rel=world_to_rel(p,o[93:96])
            qt,ok,inf=K.ik(rel,R_down(yaw),q_init=o[96:103])
            if not ok: break
        obs=step_to(env,obs,qt,chunk,grip,kp=kp,rec=rec); used+=chunk
    return obs,used
