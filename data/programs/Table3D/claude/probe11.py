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
class R:
    def __init__(s,seed=0):
        s.env=make_env(); s.obs,_=s.env.reset(seed=seed); s.q=getq(s.obs)
    def goto(s,qt,lim=0.3,maxit=60):
        rej=0
        for _ in range(maxit):
            d=qt-s.q
            if np.max(np.abs(d))<3e-3: return True
            a=np.zeros(11); a[3:10]=np.clip(d,-lim,lim)
            o2,*_=s.env.step(a); qn=getq(o2)
            if np.allclose(qn,s.q,atol=1e-9):
                rej+=1
                if rej>1: return False
            s.q=qn; s.obs=o2
        return False
def mk(yaw):
    Rt=Rz(yaw)@np.array([[1,0,0],[0,-1,0],[0,0,-1.]])
    def res(q,pos):
        T=fk(q,0.0); e=np.zeros(9); e[:3]=T[:3,3]-pos
        e[3:6]=(T[:3,0]-Rt[:,0])*0.4; e[6:9]=(T[:3,2]-Rt[:,2])*0.4
        return e
    def ikf(pos,q0):
        best=None
        for c0 in [q0,HOME]:
            s=least_squares(res,c0,args=(pos,),xtol=1e-12,ftol=1e-12,max_nfev=250)
            e=np.linalg.norm(s.fun)
            if best is None or e<best[0]: best=(e,s.x)
            if e<1e-4: break
        return best[1],best[0]
    return ikf
yaw=float(sys.argv[1])
ikf=mk(yaw)
r=R(0)
c=r.obs.get_object_from_name("cube0")
cp=np.array([float(r.obs.get(c,f)) for f in ["pose_x","pose_y","pose_z"]])-np.array([0,0,BZ])
print("yaw",yaw,"cube",cp)
for dname,dvec,start in [("+x",np.array([1.,0,0]),np.array([-0.18,0,0])),
                          ("-x",np.array([-1.,0,0]),np.array([0.18,0,0])),
                          ("+y",np.array([0,1.,0]),np.array([0,-0.18,0])),
                          ("-y",np.array([0,-1.,0]),np.array([0,0.18,0]))]:
    # descend at start point
    z=0.32; pos=cp+start+np.array([0,0,z])
    qs,e=ikf(pos,r.q)
    if e>1e-3 or not r.goto(qs): print(dname,"start unreachable"); continue
    lastok=z
    while z>0.14:
        z-=0.01
        qn,e=ikf(cp+start+np.array([0,0,z]),r.q)
        if e>1e-3: break
        if not r.goto(qn,maxit=25): break
        lastok=z
    # now slide horizontally at lastok+0.005
    zz=lastok
    p=cp+start+np.array([0,0,zz]); qn,e=ikf(p,r.q); r.goto(qn)
    d=0.0; blocked=None
    while d<0.36:
        d+=0.005
        pn=cp+start+dvec*d+np.array([0,0,zz])
        qn,e=ikf(pn,r.q)
        if e>1e-3: blocked="ikfail"; break
        if not r.goto(qn,maxit=15): blocked=round(float((start+dvec*d)@dvec),4); break
    print(dname,"floor z=%.2f"%zz,"blocked at offset along dir:",blocked, "(interface pos rel cube:",np.round(start+dvec*d,3),")",flush=True)
