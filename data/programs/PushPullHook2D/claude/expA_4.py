import numpy as np
from helper import Sim
from expA_lib import rel, robot_for
def setup(sd,gap,d=1.15,hth=np.pi/2,dy=0.30):
    s=Sim(sd); s.grasp_hook(d); o=s.obs
    Cl,dth=rel(s); M=o[20:22].copy()
    Cdes=np.array([M[0]-gap, M[1]+dy]); Rp,rth=robot_for(Cl,dth,Cdes,hth)
    a=s.goto(o[0],o[1],rth,0.2,1.0); b=s.goto(Rp[0],s.obs[1],rth,0.2,1.0); c=s.goto(Rp[0],Rp[1],rth,0.2,1.0)
    if min(a,b,c)<0: print("  WARN goto stuck",a,b,c,flush=True)
    return s
def push(s,vec,N):
    C0=s.obs[9:11].copy(); M0=s.obs[20:22].copy(); mn=9
    for i in range(N):
        s.step([vec[0],vec[1],0,0,1.0]); mn=min(mn,s.obs[20]-s.obs[9])
        if s.term: break
    o=s.obs
    return o[9:11]-C0, o[20:22]-M0, mn
print("== inc sweep, long-bar flat face, 0.4 travel ==",flush=True)
for inc in [0.03,0.04]:
    s=setup(42,0.12); dC,dM,mn=push(s,(inc,0),int(round(0.4/inc)))
    print(" inc=%.3f dC=%.4f dM=(%.4f,%.4f) gain=%.3f minsep=%.4f"%(inc,dC[0],dM[0],dM[1],dM[0]/dC[0],mn),flush=True); s.close()
print("== diagonal motion (bar vertical, normal=x) ==",flush=True)
for v in [(0.01,0.01),(0.01,-0.01),(0.005,0.02)]:
    s=setup(42,0.12); dC,dM,mn=push(s,v,20)
    print(" v=%s dC=(%.4f,%.4f) dM=(%.4f,%.4f) minsep=%.4f"%(v,dC[0],dC[1],dM[0],dM[1],mn),flush=True); s.close()
print("== tangential slide test: get sep~0.09 then move +y only ==",flush=True)
s=setup(42,0.12)
for i in range(200):
    if s.obs[20]-s.obs[9]<=0.0955: break
    s.step([0.002,0,0,0,1.0])
print(" sep=%.4f"%(s.obs[20]-s.obs[9]),flush=True)
dC,dM,mn=push(s,(0,0.01),10); print(" pure tangential dC=(%.4f,%.4f) dM=(%.4f,%.4f)"%(dC[0],dC[1],dM[0],dM[1]),flush=True); s.close()
