import numpy as np
from helper import Sim, wrap, R

def rel(s):
    o=s.obs; rth=o[2]
    Cl=R(-rth)@(o[9:11]-o[:2]); dth=wrap(o[11]-rth)
    return Cl,dth
def robot_for(Cl,dth,C_des,hth_des):
    rth=wrap(hth_des-dth); Rp=np.asarray(C_des)-R(rth)@Cl
    return Rp,rth
def bar_pts(o):
    th=o[11]; C=o[9:11].copy()
    a=np.array([-np.cos(th),-np.sin(th)]); b=np.array([np.sin(th),-np.cos(th)])
    return C,a,b
def info(s):
    o=s.obs
    return dict(R=o[:2].copy(),rth=o[2],C=o[9:11].copy(),hth=o[11],M=o[20:22].copy(),T=o[29:31].copy())
