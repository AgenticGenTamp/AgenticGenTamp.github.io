import numpy as np, sys, time
from env_client import make_env
from fk import fk
from scipy.optimize import least_squares
J=["joint_%d"%i for i in range(1,8)]
BZ=0.4; HOME=np.array([0,-0.35,-3.1416,-2.5,0,-0.87,1.5708])
def getq(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in J])
rng=np.random.default_rng(0)
def Rz(t): return np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1.]])
yaw=float(sys.argv[1]); cube=sys.argv[2] if len(sys.argv)>2 else "cube0"
seed=int(sys.argv[3]) if len(sys.argv)>3 else 0
Rt=Rz(yaw)@np.array([[1,0,0],[0,-1,0],[0,0,-1.]])
def ikf(pos,c0):
    def res(qq):
        T=fk(qq,0.0); e=np.zeros(9); e[:3]=T[:3,3]-pos
        e[3:6]=(T[:3,0]-Rt[:,0])*0.4; e[6:9]=(T[:3,2]-Rt[:,2])*0.4
        return e
    best=None
    for s0 in [c0,HOME]+[HOME+rng.normal(0,1.0,7) for _ in range(3)]:
        s=least_squares(res,s0,xtol=1e-11,ftol=1e-11,max_nfev=150); e=np.linalg.norm(s.fun)
        if best is None or e<best[0]: best=(e,s.x)
        if e<1e-5: break
    return best[1],best[0]
env=make_env(); obs,_=env.reset(seed=seed); q=getq(obs); qhome=q.copy()
c=obs.get_object_from_name(cube)
cp=np.array([float(obs.get(c,f)) for f in ["pose_x","pose_y","pose_z"]])-np.array([0,0,BZ])
def goto(qt,g=0.0,lim=0.25,maxit=50):
    global q,obs
    rej=0
    for _ in range(maxit):
        d=qt-q
        if np.max(np.abs(d))<4e-3: return True
        a=np.zeros(11); a[3:10]=np.clip(d,-lim,lim); a[10]=g
        o2,*_=env.step(a); qn=getq(o2)
        if np.allclose(qn,q,atol=1e-9):
            rej+=1
            if rej>1: return False
        q=qn; obs=o2
    return False
def grasped():
    r=obs.get_object_from_name("robot"); return float(obs.get(r,"grasp_active"))>0.5
print("yaw",yaw,cube,"seed",seed)
for c_ in np.round(np.arange(0.19,0.291,0.02),3):
    print("=== c=%.2f"%c_)
    for a_ in np.round(np.arange(0.07,0.171,0.01),3):
        row=[]
        for b_ in np.round(np.arange(-0.06,0.061,0.01),3):
            v=np.array([a_,b_,c_])
            pos = cp - Rt@v
            qt,e = ikf(pos,q)
            if e>1e-3: row.append("?"); continue
            qs,e2 = ikf(pos+np.array([0,0,0.10]),q)
            if e2<1e-3: goto(qs,g=1.0)
            if not goto(qt,g=1.0): row.append("X"); goto(qhome); continue
            aa=np.zeros(11); aa[10]=-1.0; obs,*_=env.step(aa)
            if grasped():
                row.append("G"); aa=np.zeros(11); aa[10]=1.0; obs,*_=env.step(aa)
            else: row.append(".")
            goto(qhome)
        print("  a=%.2f"%a_, " ".join(row),flush=True)
