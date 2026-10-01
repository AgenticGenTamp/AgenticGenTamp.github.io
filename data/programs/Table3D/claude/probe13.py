import numpy as np
from env_client import make_env
from fk import fk
from scipy.optimize import least_squares
J=["joint_%d"%i for i in range(1,8)]
BZ=0.4; HOME=np.array([0,-0.35,-3.1416,-2.5,0,-0.87,1.5708])
def getq(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in J])
def Rz(a): return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
Rt=np.array([[1,0,0],[0,-1,0],[0,0,-1.]])
def ikf(pos,c0):
    def res(qq):
        T=fk(qq,0.0); e=np.zeros(9); e[:3]=T[:3,3]-pos
        e[3:6]=(T[:3,0]-Rt[:,0])*0.4; e[6:9]=(T[:3,2]-Rt[:,2])*0.4
        return e
    best=None
    for s0 in [c0,HOME]:
        s=least_squares(res,s0,xtol=1e-12,ftol=1e-12,max_nfev=250); e=np.linalg.norm(s.fun)
        if best is None or e<best[0]: best=(e,s.x)
        if e<1e-4: break
    return best[1],best[0]
def report(o,tag):
    r=o.get_object_from_name("robot")
    ga=float(o.get(r,"grasp_active")); fs=float(o.get(r,"finger_state"))
    cz=[round(float(o.get(o.get_object_from_name(n),"pose_z")),3) for n in ["cube0","cube1","cube2","cube3"]]
    print(tag,"ga",ga,"fs",fs,"cubez",cz)
for OFFX in [0.06, 0.0]:
  for grip in [-1.0, 0.0]:
    env=make_env(); obs,_=env.reset(seed=0); q=getq(obs)
    c=obs.get_object_from_name("cube0")
    cp=np.array([float(obs.get(c,f)) for f in ["pose_x","pose_y","pose_z"]])-np.array([0,0,BZ])
    base=cp-np.array([OFFX,0,0])
    def goto(qt,g=0.0,lim=0.15,maxit=60):
        global q,obs
        rej=0
        for _ in range(maxit):
            d=qt-q
            if np.max(np.abs(d))<3e-3: return True
            a=np.zeros(11); a[3:10]=np.clip(d,-lim,lim); a[10]=g
            o2,*_=env.step(a); qn=getq(o2)
            if np.allclose(qn,q,atol=1e-9):
                rej+=1
                if rej>2: return False
            q=qn; obs=o2
        return False
    qs,e=ikf(base+np.array([0,0,0.32]),q); goto(qs)
    z=0.32; blocked=None
    while z>0.10:
        z-=0.01
        qn,e=ikf(base+np.array([0,0,z]),q)
        if e>1e-3: blocked="ik"; break
        if not goto(qn,g=grip,maxit=20): blocked=round(z,3); break
        r=obs.get_object_from_name("robot")
        if float(obs.get(r,"grasp_active"))>0.5 or float(obs.get(r,"finger_state"))!=0:
            print("GRASP during descent at z",z); break
    print("OFFX",OFFX,"grip",grip,"blocked at",blocked)
    report(obs,"  after descent")
    # now push harder with close flag repeatedly
    for _ in range(10):
        qn,e=ikf(base+np.array([0,0,z-0.03]),q)
        a=np.zeros(11); a[3:10]=np.clip(qn-q,-0.05,0.05); a[10]=-1.0
        o2,*_=env.step(a); q=getq(o2); obs=o2
    report(obs,"  after push+close")
    env.close()
