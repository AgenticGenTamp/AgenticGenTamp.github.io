import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,_=env.reset(seed=0)
c=kutil.Ctl(env,o)
B=c.opos("box0"); H=np.array([0.1,0.15,0.1])
# approach +y face: base at box_y + 0.15 + d, facing -y (rot=-pi/2)
d=0.45
print("drive",c.gotobase(B[0], B[1]+H[1]+d, -np.pi/2), "box",np.round(B,3))
b,q,g=c.robot(); axb,ayb=c.armbase(b)
def T(qq): return kin.fk(qq,base_x=axb,base_y=ayb,base_rot=b[2],mount=(0,0,0.245))
def gp(): 
    _,qq,_=c.robot(); return T(qq)[:3,3]
def attempt(tag,pos,R=None,yaw=None):
    if R is None: ok=c.move_to(list(pos),yaw=yaw)
    else: ok=c.move_R(list(pos),R)
    p=gp(); err=np.linalg.norm(p-np.asarray(pos))
    c.grip(True); s=c.robot()[2]
    if s>0.5:
        print("  *** GRASP",tag,"gp",np.round(p,3),"off",np.round(p-B,3))
        return True,p
    print("  %s tgt%s ok%d err%.3f gp%s"%(tag,np.round(pos,3),ok,err,np.round(p,3)))
    return False,p
def home():
    c.move_to([axb+0.25*np.cos(b[2]), ayb+0.25*np.sin(b[2]), 0.45])
# S1: top-down descend over top-face points
print("S1 top-down over box")
for (dx,dy) in [(0,0),(0,0.15),(0.1,0.15),(0.1,0),(0,0.10),(0.05,0.12)]:
    home()
    tx,ty=B[0]+dx,B[1]+dy
    c.move_to([tx,ty,0.45])
    got=None
    for z in [0.35,0.30,0.26,0.24,0.22,0.21,0.205,0.20]:
        if not c.move_to([tx,ty,z]): break
        got=z
    ok,p=attempt("t(%.2f,%.2f) lowz=%s"%(dx,dy,got),[tx,ty,got if got else 0.45])
    if ok: break
env.close()
