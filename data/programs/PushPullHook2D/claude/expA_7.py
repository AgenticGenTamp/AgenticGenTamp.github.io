import numpy as np
from helper import Sim
from expA_lib import rel, robot_for
from expA_place import setup, cand
def place_any(s,Cl,dth,u,bars=('long','short'),ts=(0.3,0.6,0.9),s0=0.11,back=0.30):
    M=s.obs[20:22].copy(); u=np.asarray(u,float); u=u/np.linalg.norm(u)
    for bar in bars:
        for t in ts:
            for hth,C in cand(u,bar,t,s0,M):
                Cs=C-u*back; Rp,rth=robot_for(Cl,dth,Cs,hth)
                if 0.16<Rp[0]<3.34 and 0.16<Rp[1]<1.10:
                    o=s.obs
                    a=s.goto(o[0],o[1],rth,0.2,1.0); b=s.goto(Rp[0],s.obs[1],rth,0.2,1.0); c=s.goto(Rp[0],Rp[1],rth,0.2,1.0)
                    return dict(bar=bar,t=t,hth=hth,Rp=Rp,err=np.linalg.norm(s.obs[:2]-Rp),st=(a,b,c))
    return None
u=None
# ---- 1) fine threshold on seed 7 ----
s,Cl,dth=setup(7)
M=s.obs[20:22].copy(); T=s.obs[29:31].copy(); u=(T-M)/np.linalg.norm(T-M)
r=place_any(s,Cl,dth,u); print("placed",r['bar'],r['t'],"err=%.3f"%r['err'],flush=True)
sep=lambda: (s.obs[20:22]-s.obs[9:11])@u
mt =lambda: np.linalg.norm(s.obs[20:22]-s.obs[29:31])
for i in range(120):
    if mt()<0.118: break
    s.step([u[0]*0.01,u[1]*0.01,0,0,1.0])
    if s.term: break
print("coarse done |MT|=%.4f sep=%.4f term=%s"%(mt(),sep(),s.term),flush=True)
hist=[]
for k in range(30):
    if s.term: break
    # retract/advance to sep ~0.1005
    for j in range(30):
        e=0.1005-sep()
        if abs(e)<1e-4: break
        st=np.clip(-e,-0.05,0.05)   # moving hook +u decreases sep
        s.step([u[0]*st,u[1]*st,0,0,1.0])
    before=mt()
    s.step([u[0]*0.0455,u[1]*0.0455,0,0,1.0])
    hist.append((before,mt(),s.term))
    if s.term: break
for b,a,t in hist[-6:]: print("   |MT| %.4f -> %.4f term=%s"%(b,a,t),flush=True)
print("THRESHOLD bracket: last_no_term=%.4f  term_at=%.4f"%(hist[-1][0],hist[-1][1]),flush=True)
s.close()
