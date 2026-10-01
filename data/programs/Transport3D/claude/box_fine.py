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
c.grip(True); print("grasp",c.robot()[2])
for z in [0.4,0.6,0.72]: c.move_to([G[0],G[1],z],yaw=YAW)
c.gotobase(0.0,0.0,0.0); c.gotobase(0.1,0.0,0.0)
for p in [[0.4,0.15,0.72],[0.55,0.15,0.70],[0.6,0.15,0.66],[0.6,0.15,0.62],[0.6,0.15,0.61]]:
    c.move_to(p,yaw=YAW)
for z in [0.605,0.6025,0.601,0.6005,0.6002,0.6001,0.60005,0.6]:
    r=c.move_to([0.6,0.15,z],yaw=YAW)
    print("z",z,r,"boxz",round(float(c.opos("box0")[2]),5))
c.grip(False); print("rel",c.robot()[2],np.round(c.opos("box0"),5))
env.close()
