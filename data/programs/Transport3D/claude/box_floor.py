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
c.move_to([G[0],G[1],0.5],yaw=YAW)
# move sideways a bit then lower back to floor
tx,ty=G[0]+0.15,G[1]
c.move_to([tx,ty,0.5],yaw=YAW)
for z in [0.4,0.3,0.25,0.22,0.21,0.205,0.2025,0.201,0.2005,0.2001,0.2]:
    r=c.move_to([tx,ty,z],yaw=YAW)
    print("z",z,r,"boxz",round(float(c.opos("box0")[2]),5))
c.grip(False); print("rel",c.robot()[2],np.round(c.opos("box0"),5))
env.close()
