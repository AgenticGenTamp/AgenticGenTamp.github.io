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

def dbg(tag):
    o=s.obs
    print("  %s R=(%.3f,%.3f) C=(%.3f,%.3f) hth=%.3f M=(%.4f,%.4f) sep=%.4f |MT|=%.4f"%(tag,o[0],o[1],o[9],o[10],o[11],o[20],o[21],sep(),mt()),flush=True)
dbg("start")
for i in range(200):
    pM=s.obs[20:22].copy(); pR=s.obs[:2].copy()
    s.step([u[0]*0.01,u[1]*0.01,0,0,1.0])
    moved=np.linalg.norm(s.obs[:2]-pR)
    dm=np.linalg.norm(s.obs[20:22]-pM)
    if dm>1e-6 or moved<1e-7:
        print("  i%3d rmoved=%.4f dM=%.4f sep_now=%.4f |MT|=%.4f term=%s R=(%.3f,%.3f)"%(i,moved,dm,sep(),mt(),s.term,s.obs[0],s.obs[1]),flush=True)
    if s.term or moved<1e-7: break
dbg("end")
print("TERM=%s |MT|=%.4f"%(s.term,mt()))
s.close()
