import numpy as np, sys, time
from env_client import make_env
from fk import fk
from scipy.optimize import least_squares
J=["joint_%d"%i for i in range(1,8)]
BZ=0.4; HOME=np.array([0,-0.35,-3.1416,-2.5,0,-0.87,1.5708])
def getq(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in J])
rng=np.random.default_rng(0)
def ikf(pos,Rt,c0,ntry=6):
    def res(qq):
        T=fk(qq,0.0); e=np.zeros(9); e[:3]=T[:3,3]-pos
        e[3:6]=(T[:3,0]-Rt[:,0])*0.4; e[6:9]=(T[:3,2]-Rt[:,2])*0.4
        return e
    best=None
    cands=[c0,HOME]+[HOME+rng.normal(0,1.5,7) for _ in range(ntry)]
    for s0 in cands:
        s=least_squares(res,s0,xtol=1e-12,ftol=1e-12,max_nfev=200); e=np.linalg.norm(s.fun)
        if best is None or e<best[0]: best=(e,s.x)
        if e<1e-5: break
    return best[1],best[0]
env=make_env(); obs,_=env.reset(seed=0); q=getq(obs); qhome=q.copy()
c=obs.get_object_from_name("cube0")
cp=np.array([float(obs.get(c,f)) for f in ["pose_x","pose_y","pose_z"]])-np.array([0,0,BZ])
def goto(qt,g=0.0,lim=0.2,maxit=70):
    global q,obs
    rej=0
    for _ in range(maxit):
        d=qt-q
        if np.max(np.abs(d))<3e-3: return True
        a=np.zeros(11); a[3:10]=np.clip(d,-lim,lim); a[10]=g
        o2,*_=env.step(a); qn=getq(o2)
        if np.allclose(qn,q,atol=1e-9):
            rej+=1
            if rej>1: return False
        q=qn; obs=o2
    return False
def check():
    r=obs.get_object_from_name("robot")
    return float(r and obs.get(r,"grasp_active")), float(obs.get(r,"finger_state"))
res=[]
t0=time.time()
for az in np.arange(0,2*np.pi,np.pi/4):
    zdir=np.array([-np.cos(az),-np.sin(az),0.])   # approach direction pointing toward cube
    xdir=np.array([-np.sin(az),np.cos(az),0.])    # finger axis horizontal
    ydir=np.cross(zdir,xdir)
    Rt=np.stack([xdir,ydir,zdir],axis=1)
    for d in [0.24,0.22,0.20,0.18,0.16,0.14,0.12]:
        pos=cp - d*zdir
        qt,e=ikf(pos,Rt,q)
        if e>1e-3: res.append((round(az,2),d,"ik")); continue
        pre=cp-(d+0.10)*zdir
        qp,e2=ikf(pre,Rt,qt)
        if e2<1e-3: goto(qp,g=1.0)
        if not goto(qt,g=1.0): res.append((round(az,2),d,"blk")); goto(qhome); continue
        for g in [-1.0,1.0]:
            a=np.zeros(11); a[10]=g; obs,*_=env.step(a)
            ga,fs=check()
            if ga>0.5 or fs!=0:
                print("GRASP az=%.2f d=%.2f g=%.1f"%(az,d,g)); sys.exit()
        res.append((round(az,2),d,"try"))
        goto(qhome)
    if time.time()-t0>500: break
from collections import Counter
print(Counter([r[2] for r in res]))
print([r for r in res if r[2]=="try"])
print("t",time.time()-t0)
