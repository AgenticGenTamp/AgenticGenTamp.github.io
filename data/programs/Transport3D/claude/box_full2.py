import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,_=env.reset(seed=1)
c=kutil.Ctl(env,o)
cu=c.opos("cube0")
ang=np.arctan2(cu[1],cu[0]); bp=cu[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
c.gotobase(bp[0],bp[1],ang)
c.move_to([cu[0],cu[1],0.3]); c.move_to([cu[0],cu[1],cu[2]]); c.grip(True)
b,_,_=c.robot(); ax,ay=c.armbase(b)
c.move_to([ax+0.30*np.cos(b[2]), ay+0.30*np.sin(b[2]), 0.6])
c.gotobase(0.15,0.0,0.0)
for pz in [0.6,0.5,0.4255]: c.move_to([0.6,-0.3,pz])
c.grip(False); print("cube",np.round(c.opos("cube0"),4))
B=c.opos("box0"); H=np.array([0.1,0.15,0.1]); YAW=0.0
G=np.array([B[0], B[1]+H[1], 0.2])
c.move_to([0.3,0.0,0.7])
print("dr",c.gotobase(B[0], B[1]+H[1]+0.45, -np.pi/2),np.round(c.robot()[0],3))
c.move_to([G[0],G[1],0.45],yaw=YAW)
for z in [0.35,0.28,0.24,0.22,0.20]:
    if not c.move_to([G[0],G[1],z],yaw=YAW): print("blk",z); break
c.grip(True); print("box grasp",c.robot()[2])
for z in [0.4,0.6,0.72]: print("lift",z,c.move_to([G[0],G[1],z],yaw=YAW))
print("dr",c.gotobase(0.0,0.0,0.0),np.round(c.robot()[0],3),"box",np.round(c.opos("box0"),3))
print("dr",c.gotobase(0.1,0.0,0.0),np.round(c.robot()[0],3))
for p in [[0.4,0.15,0.72],[0.55,0.15,0.70],[0.6,0.15,0.66],[0.6,0.15,0.63],[0.6,0.15,0.615],[0.6,0.15,0.6075],[0.6,0.15,0.6025],[0.6,0.15,0.601],[0.6,0.15,0.6]]:
    r=c.move_to(p,yaw=YAW)
    print("mv",p,r,"box",np.round(c.opos("box0"),4))
c.grip(False); print("box rel",c.robot()[2],np.round(c.opos("box0"),4),"DONE",c.done)
for i in range(3): t,tr=c.step(np.zeros(11))
print("TERM",t,"steps",c.steps)
env.close()
