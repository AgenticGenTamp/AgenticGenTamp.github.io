import numpy as np, sys, time
from env_client import make_env
from fk import fk
from scipy.optimize import least_squares
J=["joint_%d"%i for i in range(1,8)]
BZ=0.4; HOME=np.array([0,-0.35,-3.1416,-2.5,0,-0.87,1.5708])
def getq(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in J])
def Rz(a): return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
yaw=float(sys.argv[1]); Z=float(sys.argv[2])
Rt=Rz(yaw)@np.array([[1,0,0],[0,-1,0],[0,0,-1.]])
def res_f(qq,pos):
    T=fk(qq,0.0); e=np.zeros(9); e[:3]=T[:3,3]-pos
    e[3:6]=(T[:3,0]-Rt[:,0])*0.4; e[6:9]=(T[:3,2]-Rt[:,2])*0.4
    return e
def ikf(pos,c0):
    best=None
    for s0 in [c0,HOME]:
        s=least_squares(res_f,s0,args=(pos,),xtol=1e-12,ftol=1e-12,max_nfev=250)
        e=np.linalg.norm(s.fun)
        if best is None or e<best[0]: best=(e,s.x)
        if e<1e-4: break
    return best[1],best[0]
env=make_env(); obs,_=env.reset(seed=0); q=getq(obs)
c=obs.get_object_from_name("cube0")
cp=np.array([float(obs.get(c,f)) for f in ["pose_x","pose_y","pose_z"]])-np.array([0,0,BZ])
def goto(qt,lim=0.2,maxit=40):
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
g=np.round(np.arange(-0.14,0.141,0.02),3)
print("yaw",yaw,"Z",Z,"cols dy:",list(g))
SAFE=0.32
for dx in g:
    row=[]
    for dy in g:
        qs,e=ikf(cp+np.array([dx,dy,SAFE]),q)
        if e>1e-3 or not goto(qs): row.append("?"); continue
        qt,e=ikf(cp+np.array([dx,dy,Z]),q)
        if e>1e-3: row.append("?"); continue
        row.append("." if goto(qt) else "X")
    print("dx=%+.2f"%dx," ".join(row),flush=True)
env.close()
