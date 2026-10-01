import numpy as np, sys, time, itertools
from env_client import make_env
from ik2 import solve_pos_axis
from fk import fk
J=["joint_%d"%i for i in range(1,8)]
BZ=0.4
def getq(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in J])
def rotmat(axis,ang):
    axis=axis/np.linalg.norm(axis); K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3)+np.sin(ang)*K+(1-np.cos(ang))*K@K
class R:
    def __init__(s,seed=0):
        s.env=make_env(); s.obs,_=s.env.reset(seed=seed); s.q=getq(s.obs); s.home=s.q.copy()
    def goto(s,qt,lim=0.3,maxit=120):
        rej=0
        for _ in range(maxit):
            d=qt-s.q
            if np.max(np.abs(d))<3e-3: return True
            a=np.zeros(11); a[3:10]=np.clip(d,-lim,lim)
            o2,*_=s.env.step(a); qn=getq(o2)
            if np.allclose(qn,s.q,atol=1e-8):
                rej+=1
                if rej>2: return False
            s.q=qn; s.obs=o2
        return False
    def grip(s,v):
        a=np.zeros(11); a[10]=v; s.obs,*_=s.env.step(a)
        r=s.obs.get_object_from_name("robot")
        return float(s.obs.get(r,"grasp_active")),float(s.obs.get(r,"finger_state"))
r=R(0)
c=r.obs.get_object_from_name("cube0")
cp=np.array([float(r.obs.get(c,f)) for f in ["pose_x","pose_y","pose_z"]])-np.array([0,0,BZ])
r.grip(1.0)
# direction sets: approach dir = gripper z axis
dirs = {
 "down":np.array([0,0,-1.]),
 "side_-x":np.array([-1,0,0.]),
 "side_+x":np.array([1,0,0.]),
 "side_-y":np.array([0,-1,0.]),
 "side_+y":np.array([0,1,0.]),
 "diag":np.array([-0.7,0,-0.7]),
}
t0=time.time(); tried=0
res=[]
for name,zd in dirs.items():
    zd=zd/np.linalg.norm(zd)
    for off in [0.14,0.16,0.18,0.19,0.20,0.22,0.24]:
        # gripper "grasp point" at distance off along zd from interface: interface = cp - off*zd
        pos = cp - off*zd
        if pos[2] < 0.02 and name!="down": pass
        qt,pe,ae=solve_pos_axis(pos,0.0,zdir=zd,q0=r.q)
        if pe>0.005 or ae>0.03: res.append((name,off,"ikfail")); continue
        # retreat first
        pre = cp - (off+0.12)*zd
        qp,pe2,ae2=solve_pos_axis(pre,0.0,zdir=zd,q0=qt)
        if pe2<0.01: r.goto(qp)
        if not r.goto(qt): res.append((name,off,"blocked")); continue
        tried+=1
        ga,fs=r.grip(-1.0)
        if ga>0.5 or fs!=0:
            print("GRASP!",name,off,ga,fs); sys.exit()
        r.grip(1.0)
        res.append((name,off,"ok_nograsp"))
        if time.time()-t0>500: break
for x in res: print(x)
print("tried",tried,"t",round(time.time()-t0,1))
