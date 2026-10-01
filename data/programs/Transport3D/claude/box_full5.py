import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,_=env.reset(seed=1)
c=kutil.Ctl(env,o)
def prec(p,yaw=None,n=1):
    qd=c.ik(p,yaw=yaw,pos_tol=1e-7,rot_tol=1e-4,max_iters=400)
    if qd is None: return False
    return c.goto(qd,tol=1e-7,nmax=30)
cu=c.opos("cube0")
ang=np.arctan2(cu[1],cu[0]); bp=cu[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
c.gotobase(bp[0],bp[1],ang)
c.move_to([cu[0],cu[1],0.3]); c.move_to([cu[0],cu[1],cu[2]]); c.grip(True)
b,_,_=c.robot(); ax,ay=c.armbase(b)
c.move_to([ax+0.30*np.cos(b[2]), ay+0.30*np.sin(b[2]), 0.6])
c.gotobase(0.15,0.0,0.0)
for pz in [0.6,0.5,0.4255]: c.move_to([0.6,-0.3,pz])
cmd=0.4252
for it in range(5):
    prec([0.6,-0.3,cmd])
    z=float(c.opos("cube0")[2]); err=z-0.425
    print("cube it",it,"z",round(z,6),"gap",round(err,6))
    if abs(err)<3e-5: break
    cmd-=err*0.9
c.grip(False); print("cube rel",c.robot()[2],np.round(c.opos("cube0"),6))
B=c.opos("box0"); H=np.array([0.1,0.15,0.1]); YAW=0.0
G=np.array([B[0], B[1]+H[1], 0.2])
c.move_to([0.3,0.0,0.7])
c.gotobase(B[0], B[1]+H[1]+0.45, -np.pi/2)
c.move_to([G[0],G[1],0.45],yaw=YAW)
for z in [0.35,0.28,0.24,0.22,0.20]: c.move_to([G[0],G[1],z],yaw=YAW)
c.grip(True); print("box grasp",c.robot()[2])
for z in [0.4,0.6,0.72]: c.move_to([G[0],G[1],z],yaw=YAW)
c.gotobase(0.0,0.0,0.0); c.gotobase(0.1,0.0,0.0)
for p in [[0.4,0.15,0.72],[0.55,0.15,0.70],[0.6,0.15,0.66],[0.6,0.15,0.62],[0.6,0.15,0.61],[0.6,0.15,0.605]]: c.move_to(p,yaw=YAW)
for z in [0.6008,0.6004,0.6002,0.60015]:
    prec([0.6,0.15,z],yaw=YAW)
print("boxz",round(float(c.opos("box0")[2]),7))
c.grip(False); print("box rel",c.robot()[2],np.round(c.opos("box0"),6),"DONE",c.done)
for i in range(3): t,tr=c.step(np.zeros(11))
print("TERM",t)
print("up",c.move_to([0.6,0.15,0.75],yaw=YAW))
for i in range(2): t,tr=c.step(np.zeros(11))
print("TERM after retract",t)
print("dr back",c.gotobase(-0.6,0.0,0.0))
for i in range(2): t,tr=c.step(np.zeros(11))
print("TERM after drive away",t,"steps",c.steps)
print("cube",np.round(c.opos("cube0"),6),"box",np.round(c.opos("box0"),6))
env.close()
