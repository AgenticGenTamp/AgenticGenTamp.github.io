import numpy as np, sys
from env_client import make_env
from probe_fk_world import arm_from_world, RDOWN_ARM, ee_world
from ik import jacobian
np.set_printoptions(precision=4,suppress=True,linewidth=250)
def sv(env,o,p_w,steps,k,wr=1.0,grip=0.0,tol=0.002):
    best=9; used=0
    for i in range(steps):
        q=o[128:135].copy(); pa=arm_from_world(o[125:128],p_w)
        J,M=jacobian(q); ep=pa-M[:3,3]
        Rerr=RDOWN_ARM@M[:3,:3].T
        ang=np.arccos(np.clip((np.trace(Rerr)-1)/2,-1,1)); ax=np.zeros(3)
        if ang>1e-8: ax=np.array([Rerr[2,1]-Rerr[1,2],Rerr[0,2]-Rerr[2,0],Rerr[1,0]-Rerr[0,1]])/(2*np.sin(ang))*ang
        n=np.linalg.norm(ep); best=min(best,n)
        if n<tol and ang<0.03: break
        dq=J.T@np.linalg.solve(J@J.T+0.08**2*np.eye(6),np.concatenate([ep,wr*ax]))
        a=np.zeros(11); a[3:10]=np.clip(dq*k,-0.1,0.1); a[10]=grip
        o,_,_,_,_=env.step(a); o=np.asarray(o,float); used+=1
    pa=arm_from_world(o[125:128],p_w); n=float(np.linalg.norm(pa-jacobian(o[128:135])[1][:3,3]))
    return o,n,used,best
env=make_env(); o,_=env.reset(seed=0); o=np.asarray(o,float)
PTS=[[0.75,-0.15,0.62],[0.85,-0.35,0.60],[0.70,-0.05,0.65],[0.90,-0.20,0.52]]
for k in [0.4,0.8,1.2,2.0]:
    for wr in [1.0,0.3]:
        res=[]
        for p in PTS:
            o,n,u,b=sv(env,o,p,150,k,wr); res.append((round(n,4),u))
        print(f"k={k} wr={wr}: "+"  ".join(f"err={r[0]:.4f}/u={r[1]}" for r in res))
env.close()
