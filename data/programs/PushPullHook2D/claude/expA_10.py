import numpy as np
from helper import Sim
from expA_lib import rel, robot_for
from expA_place import setup
def ab(o):
    th=o[11]; return np.array([-np.cos(th),-np.sin(th)]), np.array([np.sin(th),-np.cos(th)])
def rc(s):  # button rel corner in (a,b) coords
    o=s.obs; a,b=ab(o); d=o[20:22]-o[9:11]; return np.array([d@a,d@b])
print("### A: inner-corner nest, push along (a+b)/sqrt2",flush=True)
s,Cl,dth=setup(42); o=s.obs; M=o[20:22].copy()
hth=np.pi/2; a=np.array([0.,-1.]); b=np.array([1.,0.])
u=(a+b)/np.sqrt(2)
C0=M-0.075*a-0.075*b - 0.30*u
Rp,rth=robot_for(Cl,dth,C0,hth)
s.goto(o[0],o[1],rth,0.2,1.0); s.goto(Rp[0],s.obs[1],rth,0.2,1.0); s.goto(Rp[0],Rp[1],rth,0.2,1.0)
print(" err=%.4f rel(a,b)=%s"%(np.linalg.norm(s.obs[:2]-Rp),np.round(rc(s),4)),flush=True)
H0=s.obs[9:11].copy(); M0=s.obs[20:22].copy()
for i in range(45):
    p=s.obs[9:11].copy(); s.step([u[0]*0.01,u[1]*0.01,0,0,1.0])
    if np.linalg.norm(s.obs[9:11]-p)<1e-8: print("  blocked i%d"%i,flush=True); break
    if s.term: break
dH=s.obs[9:11]-H0; dM=s.obs[20:22]-M0
print(" diag: dHook=(a%.4f,b%.4f) dM=(a%.4f,b%.4f) rel=%s"%(dH@a,dH@b,dM@a,dM@b,np.round(rc(s),4)),flush=True)
for v,lab in [(a,"+a only"),(b,"+b only"),(-u,"reverse")]:
    H0=s.obs[9:11].copy(); M0=s.obs[20:22].copy()
    for i in range(15):
        p=s.obs[9:11].copy(); s.step([v[0]*0.01,v[1]*0.01,0,0,1.0])
        if np.linalg.norm(s.obs[9:11]-p)<1e-8 or s.term: break
    dH=s.obs[9:11]-H0; dM=s.obs[20:22]-M0
    print("  %-8s dHook=(a%.4f,b%.4f) dM=(a%.4f,b%.4f) rel=%s"%(lab,dH@a,dH@b,dM@a,dM@b,np.round(rc(s),4)),flush=True)
s.close()
print("### B: SHORT-bar END-CAP push along +b (button beyond tip t=0.625)",flush=True)
s,Cl,dth=setup(42); o=s.obs; M=o[20:22].copy()
hth=np.pi/2; a=np.array([0.,-1.]); b=np.array([1.,0.])
for lat in [0.025, 0.10]:
    if s.obs[9] is None: pass
    C0=M-(0.625+0.06)*b-lat*a-0.25*b
    Rp,rth=robot_for(Cl,dth,C0,hth)
    o=s.obs; s.goto(o[0],o[1],rth,0.2,1.0); s.goto(Rp[0],s.obs[1],rth,0.2,1.0); s.goto(Rp[0],Rp[1],rth,0.2,1.0)
    print(" lat=%.3f err=%.4f rel(a,b)=%s"%(lat,np.linalg.norm(s.obs[:2]-Rp),np.round(rc(s),4)),flush=True)
    H0=s.obs[9:11].copy(); M0=s.obs[20:22].copy(); nj=0
    for i in range(60):
        pM=s.obs[20:22].copy(); pr=rc(s)
        p=s.obs[9:11].copy(); s.step([b[0]*0.005,b[1]*0.005,0,0,1.0])
        if np.linalg.norm(s.obs[9:11]-p)<1e-8: print("  blocked",flush=True); break
        d=s.obs[20:22]-pM
        if np.linalg.norm(d)>1e-6:
            print("  jump rel_pre=%s dM=(a%.4f,b%.4f)|%.4f| rel_post=%s"%(np.round(pr,4),d@a,d@b,np.linalg.norm(d),np.round(rc(s),4)),flush=True); nj+=1
            if nj>=2: break
        if s.term: break
    dH=s.obs[9:11]-H0; dM=s.obs[20:22]-M0
    print("  tot dHook=(a%.4f,b%.4f) dM=(a%.4f,b%.4f)"%(dH@a,dH@b,dM@a,dM@b),flush=True)
s.close()
