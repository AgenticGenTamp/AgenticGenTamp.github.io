import numpy as np
from helper import Sim
from expA_lib import rel, robot_for
from expA_place import setup, cand
def place_any(s,Cl,dth,u,bars=('long','short'),ts=(0.3,),s0=0.11,back=0.30):
    M=s.obs[20:22].copy(); u=np.asarray(u,float); u=u/np.linalg.norm(u)
    for bar in bars:
        for t in ts:
            for hth,C in cand(u,bar,t,s0,M):
                Cs=C-u*back; Rp,rth=robot_for(Cl,dth,Cs,hth)
                if 0.16<Rp[0]<3.34 and 0.16<Rp[1]<1.10:
                    o=s.obs
                    s.goto(o[0],o[1],rth,0.2,1.0); s.goto(Rp[0],s.obs[1],rth,0.2,1.0); s.goto(Rp[0],Rp[1],rth,0.2,1.0)
                    return dict(bar=bar,t=t,hth=hth,err=np.linalg.norm(s.obs[:2]-Rp))
    return None
def qfun(s,u,bar):
    """surface-distance q = true_sep-0.025 ; true_sep = (M-C).u -/+ 0.025 offset"""
    o=s.obs; th=o[11]
    a=np.array([-np.cos(th),-np.sin(th)]); b=np.array([np.sin(th),-np.cos(th)])
    off = 0.025*b if bar=='long' else 0.025*a
    cl = o[9:11]+off
    return (o[20:22]-cl)@u - 0.025
def moveto_q(s,u,bar,qt,maxit=60):
    for j in range(maxit):
        e=qfun(s,u,bar)-qt
        if abs(e)<1e-5: return True
        h=np.clip(e,-0.05,0.05)
        p=s.obs[9:11].copy()
        s.step([u[0]*h,u[1]*h,0,0,1.0])
        if np.linalg.norm(s.obs[9:11]-p)<1e-8: return False
    return False
print("### T1: fine push, seed7, find |M-T| termination threshold",flush=True)
s,Cl,dth=setup(7); M=s.obs[20:22].copy(); T=s.obs[29:31].copy(); u=(T-M)/np.linalg.norm(T-M)
r=place_any(s,Cl,dth,u); print(" placed",r,flush=True)
mt=lambda: np.linalg.norm(s.obs[20:22]-s.obs[29:31])
for i in range(200):
    if mt()<0.115 or s.term: break
    p=s.obs[9:11].copy(); s.step([u[0]*0.01,u[1]*0.01,0,0,1.0])
    if np.linalg.norm(s.obs[9:11]-p)<1e-8: print(" BLOCKED",flush=True); break
print(" coarse |MT|=%.4f q=%.4f term=%s"%(mt(),qfun(s,u,r['bar']),s.term),flush=True)
for k in range(40):
    if s.term: break
    if not moveto_q(s,u,r['bar'],0.0505): print("  stuck",flush=True); break
    b4=mt(); pM=s.obs[20:22].copy()
    s.step([u[0]*0.048,u[1]*0.048,0,0,1.0])
    print("  push dM=%.4f |MT| %.5f -> %.5f term=%s"%(np.linalg.norm(s.obs[20:22]-pM),b4,mt(),s.term),flush=True)
    if s.term: break
s.close()
print("### T2: long bar, push along -b (other face). predict onset q=0.05 <=> (M-C).u = 0.075",flush=True)
s,Cl,dth=setup(42); M=s.obs[20:22].copy()
hth=np.pi/2  # a=(0,-1) b=(1,0); push u=-b=(-1,0): button on -x side
u=np.array([-1.0,0.0])
Cdes=M-np.array([-0.5,0.0])-np.array([0.,-0.30])  # corner right of button, bar down thru button height
Cdes=np.array([M[0]+0.35, M[1]+0.30])
Rp,rth=robot_for(Cl,dth,Cdes,hth)
o=s.obs; s.goto(o[0],o[1],rth,0.2,1.0); s.goto(Rp[0],s.obs[1],rth,0.2,1.0); s.goto(Rp[0],Rp[1],rth,0.2,1.0)
print(" robot err=%.3f rawsep=(M-C).u=%.4f q=%.4f"%(np.linalg.norm(s.obs[:2]-Rp),(s.obs[20:22]-s.obs[9:11])@u,qfun(s,u,'long')),flush=True)
n=0
for i in range(200):
    pM=s.obs[20:22].copy(); pq=qfun(s,u,'long'); praw=(s.obs[20:22]-s.obs[9:11])@u
    p=s.obs[9:11].copy(); s.step([u[0]*0.002,u[1]*0.002,0,0,1.0])
    if np.linalg.norm(s.obs[9:11]-p)<1e-8: print(" BLOCKED",flush=True); break
    d=np.linalg.norm(s.obs[20:22]-pM)
    if d>1e-6:
        print("  contact: raw_pre=%.4f q_pre=%.4f jump=%.4f q_post=%.4f"%(praw-0.002,pq-0.002,d,qfun(s,u,'long')),flush=True); n+=1
        if n>=3: break
s.close()
