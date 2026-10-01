import numpy as np, json, sys
from env_client import make_env
import kinova_fk as K

def R_down(yaw=0.0):
    c,s=np.cos(yaw),np.sin(yaw)
    Rz=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    return Rz@np.array([[1,0,0],[0,-1,0],[0,0,-1.]])

def goto_q(env,qt,steps=60,grip=0.0,kp=3.0,tol=0.01,log=None):
    qt=np.asarray(qt,dtype=float); obs=None; r=0
    for i in range(steps):
        a=np.zeros(11,dtype=np.float32); a[10]=grip
        if obs is not None:
            q=np.asarray(obs)[96:103]
            err=qt-q
            if np.max(np.abs(err))<tol: 
                a[3:10]=np.clip(kp*err,-0.1,0.1)
            else:
                a[3:10]=np.clip(kp*err,-0.1,0.1)
        obs,r,te,tr,inf=env.step(a)
        if log is not None: log.append(np.asarray(obs).copy())
    return np.asarray(obs)

def ik_down(p, q_init, yaw=0.0):
    q,info=K.ik(np.asarray(p), R_down(yaw), q_init=np.asarray(q_init)), None
    return q

if __name__=="__main__":
    env=make_env()
    obs,info=env.reset(seed=0)
    o=np.asarray(obs); q=o[96:103].copy()
    print("base",o[93:96])
    res=K.ik(np.array([0.6,0.0,0.3]), R_down(), q_init=q)
    print("ik result type", type(res))
    print(res)
    env.close()

def scan_z(seed=0):
    env=make_env(); obs,info=env.reset(seed=seed)
    o=np.asarray(obs); q=o[96:103].copy()
    objs0=o[[0,1,2,16,17,18,32,33,34]].copy()
    zs=np.arange(0.40,-0.31,-0.04)
    for z in zs:
        qt,ok,inf=K.ik(np.array([0.6,0.0,z]), R_down(), q_init=q)
        if not ok: print("ik fail",z); continue
        o=goto_q(env,qt,steps=40)
        q=o[96:103].copy()
        err=np.max(np.abs(qt-q))
        objs=o[[0,1,2,16,17,18,32,33,34]]
        print("z=%.2f err=%.4f objmove=%.4f"%(z,err,np.max(np.abs(objs-objs0))))
    env.close()

def move_line(env,obs,p0,p1,n=20,steps_per=5,grip=0.0,yaw=0.0,rec=None):
    q=np.asarray(obs)[96:103].copy()
    for t in np.linspace(0,1,n+1):
        p=(1-t)*np.asarray(p0)+t*np.asarray(p1)
        qt,ok,_=K.ik(p,R_down(yaw),q_init=q)
        obs=goto_q(env,qt,steps=steps_per,grip=grip)
        q=np.asarray(obs)[96:103].copy()
        if rec is not None: rec.append((p.copy(),np.asarray(obs).copy()))
    return obs

def sweep(seed=0):
    import numpy as np
    env=make_env(); obs,info=env.reset(seed=seed)
    o=np.asarray(obs); print("base",np.round(o[93:96],3))
    print("bowl",np.round(o[0:3],3),"drink",np.round(o[16:19],3),"can",np.round(o[32:35],3))
    idx=[0,1,16,17,32,33]
    ref=o[idx].copy()
    rec=[]
    # go to start pose high then down
    obs=move_line(env,obs,[0.55,-0.60,0.30],[0.55,-0.60,0.06],n=4,steps_per=8)
    obs=move_line(env,obs,[0.55,-0.60,0.06],[0.55,0.45,0.06],n=21,steps_per=7,rec=rec)
    for p,ob in rec:
        d=ob[idx]-ref
        print("p_y=%.3f"%p[1], "objdel", np.round(d,3), "q_err_ok")
    env.close()

def snap(seed=0, targets=((0.55,0.0,0.30),(0.55,0.0,0.06))):
    env=make_env(); obs,info=env.reset(seed=seed)
    out=[]
    for p in targets:
        obs=move_line(env,obs,np.asarray(p)+np.array([0,0,0.0]),p,n=1,steps_per=40)
        o=np.asarray(obs)
        q=o[96:103]
        print(p,"achieved fk",np.round(K.fk(q)[:3,3],3),"qerr")
        out.append(o.tolist())
    json.dump(out,open("snap.json","w"))
    env.close()
