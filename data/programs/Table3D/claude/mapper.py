import numpy as np, sys, time
from env_client import make_env
from ik2 import solve_pos_axis
J=["joint_%d"%i for i in range(1,8)]
BZ=0.4
def getq(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in J])
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
yaw=float(sys.argv[1]) if len(sys.argv)>1 else 0.0
r=R(0)
c=r.obs.get_object_from_name("cube0")
cp=np.array([float(r.obs.get(c,f)) for f in ["pose_x","pose_y","pose_z"]])-np.array([0,0,BZ])
zd=np.array([0,0,-1.])
# yaw rotation applied via extra constraint? use solve with zdir only (yaw free) -> can't control yaw.
# Instead: full-orientation IK
from scipy.optimize import least_squares
from fk import fk
def Rz(a): return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
Rt = Rz(yaw)@np.array([[1,0,0],[0,-1,0],[0,0,-1.]])
def res(q,pos):
    T=fk(q,0.0)
    e=np.zeros(9); e[:3]=T[:3,3]-pos
    e[3:6]=(T[:3,0]-Rt[:,0])*0.4; e[6:9]=(T[:3,2]-Rt[:,2])*0.4
    return e
def ikfull(pos,q0):
    best=None
    for c0 in [q0, np.array([0,-0.35,-3.1416,-2.5,0,-0.87,1.5708])]:
        s=least_squares(res,c0,args=(pos,),xtol=1e-12,ftol=1e-12,max_nfev=200)
        e=np.linalg.norm(s.fun)
        if best is None or e<best[0]: best=(e,s.x)
        if e<1e-4: break
    return best[1],best[0]
print("yaw",yaw)
grid=[-0.06,-0.04,-0.02,0.0,0.02,0.04,0.06]
out={}
t0=time.time()
for dx in grid:
    row=[]
    for dy in grid:
        # start high
        z=0.30
        qs,e=ikfull(cp+np.array([dx,dy,z]),r.q)
        if e>1e-3 or not r.goto(qs): row.append(None); continue
        lastok=z
        while z>0.15:
            z-=0.01
            qn,e=ikfull(cp+np.array([dx,dy,z]),r.q)
            if e>1e-3: break
            if not r.goto(qn,maxit=20): break
            lastok=z
        row.append(round(lastok,3))
        # retreat
        qs,e=ikfull(cp+np.array([dx,dy,0.30]),r.q); r.goto(qs)
    print("dx=%.2f"%dx, row, flush=True)
print("time",time.time()-t0)
