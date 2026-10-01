import numpy as np, itertools, sys, time
from env_client import make_env
from fk import fk
from ik2 import solve_pos_axis
J=["joint_%d"%i for i in range(1,8)]
BZ=0.4
def getq(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in J])
class Runner:
    def __init__(self,seed=0):
        self.env=make_env(); self.obs,_=self.env.reset(seed=seed)
        self.q=getq(self.obs)
    def goto(self,qt,lim=0.25,maxit=100):
        rej=0
        for _ in range(maxit):
            d=qt-self.q
            if np.max(np.abs(d))<2e-3: return True
            a=np.zeros(11); a[3:10]=np.clip(d,-lim,lim)
            o2,*_=self.env.step(a); qn=getq(o2)
            if np.allclose(qn,self.q,atol=1e-8):
                rej+=1
                if rej>2: return False
            self.q=qn; self.obs=o2
        return False
    def grip(self,v):
        a=np.zeros(11); a[10]=v
        self.obs,*_=self.env.step(a)
        r=self.obs.get_object_from_name("robot")
        return float(self.obs.get(r,"grasp_active")), float(self.obs.get(r,"finger_state"))

r=Runner(0)
c=r.obs.get_object_from_name("cube0")
cp=np.array([float(r.obs.get(c,f)) for f in ["pose_x","pose_y","pose_z"]])-np.array([0,0,BZ])
print("cube rel",cp)
r.grip(1.0)
best=None
t0=time.time()
for dz in [0.24,0.22,0.21,0.20,0.19]:
  for yaw in [0.0, np.pi/2]:
    for dx,dy in itertools.product([-0.04,-0.02,0,0.02,0.04],repeat=2):
        tgt=cp+np.array([dx,dy,dz])
        qt,pe,ae=solve_pos_axis(tgt,0.0,zdir=np.array([0,0,-1.]),q0=r.q)
        if pe>0.004 or ae>0.02: continue
        # go to a safe height above first
        qs,_,_=solve_pos_axis(tgt+np.array([0,0,0.12]),0.0,q0=r.q)
        if not r.goto(qs): continue
        if not r.goto(qt): continue
        ga,fs=r.grip(-1.0)
        if ga>0.5 or fs!=0:
            print("GRASP!! dz=%.2f yaw=%.2f dx=%.2f dy=%.2f ga=%s fs=%s"%(dz,yaw,dx,dy,ga,fs)); sys.exit()
        r.grip(1.0)
print("no grasp found, elapsed",time.time()-t0)
