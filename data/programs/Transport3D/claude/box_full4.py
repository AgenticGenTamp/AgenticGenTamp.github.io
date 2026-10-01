import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,_=env.reset(seed=1)
c=kutil.Ctl(env,o)
cu=c.opos("cube0")
ang=np.arctan2(cu[1],cu[0]); bp=cu[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
c.gotobase(bp[0],bp[1],ang)
c.move_to([cu[0],cu[1],0.3])
print("topgrasp move",c.move_to([cu[0],cu[1],cu[2]+0.025]))
c.grip(True); print("cube grasp",c.robot()[2])
b,_,_=c.robot(); ax,ay=c.armbase(b)
c.move_to([ax+0.30*np.cos(b[2]), ay+0.30*np.sin(b[2]), 0.6])
c.gotobase(0.15,0.0,0.0)
for pz in [0.6,0.5,0.46]: c.move_to([0.6,-0.3,pz])
cmd=0.4501
for it in range(4):
    r=c.move_to([0.6,-0.3,cmd],tol=1e-7,gtol=1e-7,rot_tol=1e-4,max_iters=400)
    z=float(c.opos("cube0")[2]); err=z-0.4250
    print("it",it,r,"cubez",round(z,6),"err",round(err,6))
    if abs(err)<2e-5: break
    cmd-=err
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
for z in [0.6008,0.6004,0.6002]:
    qd=c.ik([0.6,0.15,z],yaw=YAW,pos_tol=1e-7,rot_tol=1e-4,max_iters=400)
    if qd is not None: c.goto(qd,tol=1e-7,nmax=30)
    print("z",z,"boxz",round(float(c.opos("box0")[2]),7))
c.grip(False); print("box rel",c.robot()[2],np.round(c.opos("box0"),6),"DONE",c.done)
for i in range(3): t,tr=c.step(np.zeros(11))
print("TERM",t,"steps",c.steps,"cube",np.round(c.opos("cube0"),6))
env.close()
