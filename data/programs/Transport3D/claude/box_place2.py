import numpy as np, kutil, kin, sys
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,_=env.reset(seed=0)
c=kutil.Ctl(env,o)
B=c.opos("box0"); H=np.array([0.1,0.15,0.1])
G=np.array([B[0], B[1]+H[1], 0.2]); YAW=0.0
c.gotobase(B[0], B[1]+H[1]+0.45, -np.pi/2)
c.move_to([G[0],G[1],0.45],yaw=YAW)
for z in [0.35,0.28,0.24,0.22,0.20]:
    if not c.move_to([G[0],G[1],z],yaw=YAW): print("blk",z); break
c.grip(True); print("grasp",c.robot()[2])
c.move_to([G[0],G[1],0.55],yaw=YAW)
for bx,by,brot in [(0.0,0.0,0.0),(0.1,0.0,0.0),(0.15,0.0,0.0)]:
    print("dr",bx,c.gotobase(bx,by,brot),np.round(c.robot()[0],3))
for p in [[0.35,0.15,0.7],[0.5,0.15,0.68],[0.6,0.15,0.65],[0.6,0.15,0.62],[0.6,0.15,0.61],[0.6,0.15,0.6]]:
    print("mv",p,c.move_to(p,yaw=YAW),"box",np.round(c.opos("box0"),4))
c.grip(False); print("rel",c.robot()[2],"box",np.round(c.opos("box0"),4))
t,_=c.step(np.zeros(11)); print("term",t,"steps",c.steps)
env.close()
