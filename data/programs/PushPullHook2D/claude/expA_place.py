"""General placement of grasped hook so a chosen bar face pushes M along u."""
import numpy as np
from helper import Sim, wrap, R
from expA_lib import rel, robot_for

def setup(sd,d=1.15):
    s=Sim(sd); s.grasp_hook(d); Cl,dth=rel(s); return s,Cl,dth

def cand(u,bar,t,s0,M):
    out=[]
    ux,uy=u
    if bar=='long':
        for hth in (np.arctan2(-ux,uy), np.arctan2(ux,-uy)):
            a=np.array([-np.cos(hth),-np.sin(hth)])
            out.append((hth, M-a*t-u*s0))
    else:
        for hth in (np.arctan2(-uy,-ux), np.arctan2(uy,ux)):
            b=np.array([np.sin(hth),-np.cos(hth)])
            out.append((hth, M-b*t-u*s0))
    return out

def place(s,Cl,dth,u,bar,t,s0,back=0.25,verbose=True):
    """back: extra retreat along -u before approaching"""
    M=s.obs[20:22].copy(); u=np.asarray(u,float); u=u/np.linalg.norm(u)
    best=None
    for hth,C in cand(u,bar,t,s0,M):
        Cs=C-u*back
        Rp,rth=robot_for(Cl,dth,Cs,hth)
        ok = 0.15<Rp[0]<3.35 and 0.15<Rp[1]<1.10
        if verbose: print("   cand hth=%.3f robot=(%.3f,%.3f) ok=%s"%(hth,Rp[0],Rp[1],ok))
        if ok and best is None: best=(hth,Cs,Rp,rth)
    if best is None: return None
    hth,Cs,Rp,rth=best
    o=s.obs
    a=s.goto(o[0],o[1],rth,0.2,1.0)
    b=s.goto(Rp[0],s.obs[1],rth,0.2,1.0)
    c=s.goto(Rp[0],Rp[1],rth,0.2,1.0)
    if min(a,b,c)<0 and verbose: print("   goto stuck",a,b,c)
    # approach along u until contact zone
    return hth
