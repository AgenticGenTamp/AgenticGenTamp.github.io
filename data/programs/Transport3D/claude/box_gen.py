import numpy as np, kutil, kin, sys
from env_client import make_env
kutil.MZ=0.269
seed=int(sys.argv[1])
env=make_env(); o,_=env.reset(seed=seed)
c=kutil.Ctl(env,o)
def drive(tx,ty,trot):
    b,_,_=c.robot()
    if c.gotobase(tx,ty,trot): return True
    b,_,_=c.robot()
    c.gotobase(b[0],ty,trot); 
    if c.gotobase(tx,ty,trot): return True
    b,_,_=c.robot(); c.gotobase(tx,b[1],trot)
    return c.gotobase(tx,ty,trot)
def prec(p,yaw=None):
    qd=c.ik(p,yaw=yaw,pos_tol=1e-7,rot_tol=1e-4,max_iters=400)
    if qd is None: return False
    return c.goto(qd,tol=1e-7,nmax=30)
names=sorted(n for n in c.o.get_object_names() if n.startswith("cube"))
slots=[(0.6,-0.30),(0.6,0.30),(0.45,-0.30),(0.45,0.30),(0.6,-0.38),(0.6,0.38)]
for i,n in enumerate(names):
    cu=c.opos(n)
    ang=np.arctan2(cu[1],cu[0]); bp=cu[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
    drive(bp[0],bp[1],ang)
    c.move_to([cu[0],cu[1],0.3]); c.move_to([cu[0],cu[1],cu[2]]); c.grip(True)
    g=c.robot()[2]
    b,_,_=c.robot(); ax,ay=c.armbase(b)
    c.move_to([ax+0.30*np.cos(b[2]), ay+0.30*np.sin(b[2]), 0.6])
    drive(0.15,0.0,0.0)
    sx,sy=slots[i]
    for pz in [0.6,0.5,0.4255]: c.move_to([sx,sy,pz])
    prec([sx,sy,0.4252])
    c.grip(False)
    print(n,"grasp",g,"placed",np.round(c.opos(n),4),"rel",c.robot()[2])
B=c.opos("box0"); H=np.array([float(c.o.get(c.o.get_object_from_name("box0"),f)) for f in ["half_extent_x","half_extent_y","half_extent_z"]])
b,_,_=c.robot()
# choose face: axis of largest separation from current base
d=B[:2]-b[:2]
if abs(d[0])>abs(d[1]): axis,sgn=0,-np.sign(d[0])
else: axis,sgn=1,-np.sign(d[1])
off=np.zeros(3); off[axis]=sgn*H[axis]; off[2]=H[2]
G=B+off
sd=B[:2]+off[:2]/max(H[axis],1e-9)*(H[axis]+0.45)
face_ang=np.arctan2(-off[1],-off[0])
YAW=0.0
c.move_to([b[0]+0.3*np.cos(b[2]),b[1]+0.3*np.sin(b[2]),0.7])
print("dr",drive(sd[0],sd[1],face_ang),np.round(c.robot()[0],3))
c.move_to([G[0],G[1],0.45],yaw=YAW)
for z in [0.35,0.28,0.24,0.22,0.20]:
    if not c.move_to([G[0],G[1],z],yaw=YAW): print("blk",z)
c.grip(True); print("box grasp",c.robot()[2],"gp-box off",np.round(off,3))
for z in [0.4,0.6,0.72]: c.move_to([G[0],G[1],z],yaw=YAW)
drive(0.0,0.0,0.0); c.gotobase(0.1,0.0,0.0)
T=np.array([0.6,0.0,0.5])+off   # target grasp point for box center (0.6,0,0.5)
for p in [[T[0],T[1],0.72],[T[0],T[1],0.66],[T[0],T[1],0.62],[T[0],T[1],T[2]+0.008],[T[0],T[1],T[2]+0.004]]:
    r=c.move_to(list(p),yaw=YAW)
for z in [0.0008,0.0004,0.0002,0.00015]: prec([T[0],T[1],T[2]+z],yaw=YAW)
print("boxz",np.round(c.opos("box0"),5))
c.grip(False); print("box rel",c.robot()[2])
c.move_to([T[0],T[1],0.75],yaw=YAW)
t=False
for i in range(2): t,tr=c.step(np.zeros(11))
print("TERM",t,"steps",c.steps)
env.close()
