import numpy as np, sys
from env_client import make_env
from fk import fk
from scipy.optimize import least_squares
J=["joint_%d"%i for i in range(1,8)]
BZ=0.4
HOME=np.array([0,-0.35,-3.1416,-2.5,0,-0.87,1.5708])
def getq(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in J])
def Rz(a): return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
env=make_env(); obs,_=env.reset(seed=0); q0=getq(obs)
c=obs.get_object_from_name("cube0")
cp=np.array([float(obs.get(c,f)) for f in ["pose_x","pose_y","pose_z"]])-np.array([0,0,BZ])
q=q0.copy()
def goto(qt,lim=0.2,maxit=80):
    global q,obs
    rej=0
    for _ in range(maxit):
        d=qt-q
        if np.max(np.abs(d))<3e-3: return True
        a=np.zeros(11); a[3:10]=np.clip(d,-lim,lim)
        o2,*_=env.step(a); qn=getq(o2)
        if np.allclose(qn,q,atol=1e-9):
            rej+=1
            if rej>1: return False
        q=qn; obs=o2
    return False
rng=np.random.default_rng(1)
def ikf(pos,yaw,c0):
    Rt=Rz(yaw)@np.array([[1,0,0],[0,-1,0],[0,0,-1.]])
    def res(qq):
        T=fk(qq,0.0); e=np.zeros(9); e[:3]=T[:3,3]-pos
        e[3:6]=(T[:3,0]-Rt[:,0])*0.4; e[6:9]=(T[:3,2]-Rt[:,2])*0.4
        return e
    s=least_squares(res,c0,xtol=1e-12,ftol=1e-12,max_nfev=300)
    return s.x, np.linalg.norm(s.fun)
for dz in [0.20,0.18,0.16,0.14,0.12,0.10]:
    ok=0; tot=0; sols=[]
    for k in range(25):
        yaw=rng.uniform(0,2*np.pi)
        c0=HOME+rng.normal(0,1.5,7)
        qs,e=ikf(cp+np.array([0,0,dz]),yaw,c0)
        if e>1e-3: continue
        tot+=1
        if goto(qs):
            ok+=1; sols.append((yaw,np.round(qs,2)))
        goto(q0)  # return home
    print("dz",dz,"reachable",ok,"/",tot,flush=True)
    if sols: print("   e.g.",sols[0])
env.close()
