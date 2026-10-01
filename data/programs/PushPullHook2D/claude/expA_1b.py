import sys, numpy as np
from helper import Sim, wrap, R
from expA_lib import rel, robot_for
def run(sd,inc,gap,N,quiet=True):
    s=Sim(sd); s.grasp_hook(1.15); o=s.obs
    Cl,dth=rel(s); M=o[20:22].copy()
    hth=np.pi/2; Cdes=np.array([M[0]-gap, M[1]+0.30])
    Rp,rth=robot_for(Cl,dth,Cdes,hth)
    if s.goto(o[0],o[1],rth,0.2,1.0)<0: print("rotfail")
    s.goto(Rp[0],s.obs[1],rth,0.2,1.0); s.goto(Rp[0],Rp[1],rth,0.2,1.0)
    o=s.obs; M0=o[20:22].copy(); C0=o[9:11].copy(); n0=s.n
    seps=[]; jumps=[]; first=None
    for i in range(N):
        p=s.obs[[9,10,20,21]].copy()
        s.step([inc,0,0,0,1.0]); o=s.obs
        dM=o[20]-p[2+0]
        sep=o[20]-o[9]
        if dM>1e-6:
            if first is None: first=(p[2]-p[0], sep, dM)  # sep_before, sep_after, jump
            jumps.append(dM)
        seps.append(sep)
        if s.term: break
    o=s.obs
    print("sd%d inc=%.3f gap=%.2f: sep_range=[%.4f,%.4f] njump=%d jump=%s totdC=%.4f totdM=(%.4f,%.4f) gain=%.3f term=%s"%(
        sd,inc,gap,min(seps),max(seps),len(jumps),
        ("%.4f..%.4f"%(min(jumps),max(jumps))) if jumps else "-",
        o[9]-C0[0],o[20]-M0[0],o[21]-M0[1],(o[20]-M0[0])/max(1e-9,o[9]-C0[0]),s.term))
    if first: print("   first contact: sep_before=%.4f -> sep_after=%.4f jump=%.4f"%first)
    s.close()
for inc in [0.005,0.01,0.02,0.05]:
    run(42,inc,0.12,int(round(0.4/inc)))
