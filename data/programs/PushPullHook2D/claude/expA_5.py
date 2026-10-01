import numpy as np, sys
from helper import Sim
from expA_place import setup, place
def approach_push(s,u,inc,N,label):
    u=np.asarray(u,float); u/=np.linalg.norm(u)
    M0=s.obs[20:22].copy(); C0=s.obs[9:11].copy(); first=None; k=0
    for i in range(N):
        p=s.obs[20:22].copy()
        s.step([u[0]*inc,u[1]*inc,0,0,1.0]); k+=1
        d=s.obs[20:22]-p
        if np.linalg.norm(d)>1e-6 and first is None: first=(i,d.copy())
        if s.term: break
    dC=s.obs[9:11]-C0; dM=s.obs[20:22]-M0
    perp=np.array([-u[1],u[0]])
    print(" %-22s dC=%.4f dM_par=%.4f dM_perp=%+.5f gain=%.3f steps=%d firstjump=%s"%(
        label,dC@u,dM@u,dM@perp,(dM@u)/max(1e-9,dC@u),k, "None" if first is None else "i%d %s"%(first[0],np.round(first[1],4))))
    return dM
sd=int(sys.argv[1]) if len(sys.argv)>1 else 42
print("== SHORT bar face push (t=0.3 along short bar), u=(1,0) ==")
s,Cl,dth=setup(sd); u=(1,0)
if place(s,Cl,dth,u,'short',0.30,0.11,back=0.30):
    print("  sep0=%.4f"%np.linalg.norm(s.obs[20:22]-s.obs[9:11]))
    approach_push(s,u,0.01,60,"short face inc.01")
s.close()
print("== SHORT bar face push u=(0,-1) ==")
s,Cl,dth=setup(sd); u=(0,-1)
if place(s,Cl,dth,u,'short',0.30,0.11,back=0.30): approach_push(s,u,0.01,60,"short face -y")
s.close()
print("== LONG bar TIP push: contact at t=1.24 (near end), u along bar normal ==")
s,Cl,dth=setup(sd); u=(1,0)
if place(s,Cl,dth,u,'long',1.24,0.11,back=0.30): approach_push(s,u,0.01,60,"long t=1.24")
s.close()
print("== LONG bar beyond end t=1.30 (should MISS) ==")
s,Cl,dth=setup(sd); u=(1,0)
if place(s,Cl,dth,u,'long',1.30,0.11,back=0.30): approach_push(s,u,0.01,60,"long t=1.30")
s.close()
