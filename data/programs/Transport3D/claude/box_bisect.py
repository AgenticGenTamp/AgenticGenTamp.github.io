import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,_=env.reset(seed=0)
c=kutil.Ctl(env,o)
B=c.opos("box0"); H=np.array([0.1,0.15,0.1]); YAW=0.0
G=np.array([B[0], B[1]+H[1], 0.2])
c.gotobase(B[0], B[1]+H[1]+0.45, -np.pi/2)
c.move_to([G[0],G[1],0.45],yaw=YAW)
for z in [0.35,0.28,0.24,0.22,0.20]: c.move_to([G[0],G[1],z],yaw=YAW)
c.grip(True)
for z in [0.4,0.6,0.72]: c.move_to([G[0],G[1],z],yaw=YAW)
c.gotobase(0.0,0.0,0.0); c.gotobase(0.1,0.0,0.0)
for p in [[0.4,0.15,0.72],[0.55,0.15,0.70],[0.6,0.15,0.66],[0.6,0.15,0.62],[0.6,0.15,0.61],[0.6,0.15,0.605]]: c.move_to(p,yaw=YAW)
def prec(z):
    qd=c.ik([0.6,0.15,z],yaw=YAW,pos_tol=1e-7,rot_tol=1e-4,max_iters=400,restarts=6)
    if qd is None: return "IKfail",None
    ok=c.goto(qd,tol=1e-7,nmax=30)
    return ("goto%d"%ok), float(c.opos("box0")[2])
for z in [0.6015,0.6008,0.6004,0.6002,0.6001,0.60005,0.60002,0.60001,0.6,0.59999]:
    s,bz=prec(z); print(z,s,None if bz is None else round(bz,7))
env.close()
