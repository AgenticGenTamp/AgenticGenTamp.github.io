import numpy as np, itertools, sys, time
from env_client import make_env
from ik2 import solve_pos_axis
J=["joint_%d"%i for i in range(1,8)]
BZ=0.4
def getq(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in J])
class R:
    def __init__(s,seed=0):
        s.env=make_env(); s.obs,_=s.env.reset(seed=seed); s.q=getq(s.obs)
    def goto(s,qt,lim=0.25,maxit=120):
        rej=0
        for _ in range(maxit):
            d=qt-s.q
            if np.max(np.abs(d))<2e-3: return True
            a=np.zeros(11); a[3:10]=np.clip(d,-lim,lim)
            o2,*_=s.env.step(a); qn=getq(o2)
            if np.allclose(qn,s.q,atol=1e-8):
                rej+=1
                if rej>2: return False
            s.q=qn; s.obs=o2
        return False
    def grip(s,v):
        a=np.zeros(11); a[10]=v; s.obs,*_=s.env.step(a)
        r=s.obs.get_object_from_name("robot")
        return float(s.obs.get(r,"grasp_active")),float(s.obs.get(r,"finger_state"))
r=R(0)
c=r.obs.get_object_from_name("cube0")
cp=np.array([float(r.obs.get(c,f)) for f in ["pose_x","pose_y","pose_z"]])-np.array([0,0,BZ])
r.grip(1.0)
tried=0; blocked=0; ikfail=0
t0=time.time()
for dz in [0.35,0.30,0.27,0.25,0.23,0.22,0.21]:
  for yaw in [0.0,np.pi/4,np.pi/2]:
    zd=np.array([0,0,-1.])
    for dx,dy in [(0,0),(-0.03,0),(0.03,0),(0,-0.03),(0,0.03),(-0.06,0),(0.06,0),(0,-0.06),(0,0.06)]:
        if time.time()-t0>500: break
        tgt=cp+np.array([dx,dy,dz])
        qt,pe,ae=solve_pos_axis(tgt,0.0,zdir=zd,q0=r.q)
        if pe>0.004 or ae>0.02: ikfail+=1; continue
        if not r.goto(qt): blocked+=1; continue
        tried+=1
        ga,fs=r.grip(-1.0)
        if ga>0.5 or fs!=0:
            print("GRASP dz=%.2f dx=%.2f dy=%.2f ga=%s fs=%s"%(dz,dx,dy,ga,fs)); sys.exit()
        r.grip(1.0)
print("no grasp. tried",tried,"blocked",blocked,"ikfail",ikfail,"t",time.time()-t0)
