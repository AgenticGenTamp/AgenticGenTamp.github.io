import numpy as np, kutil, kin, sys
from env_client import make_env
kutil.MZ=0.269
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
YAW=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
env=make_env(); o,_=env.reset(seed=seed)
c=kutil.Ctl(env,o)
B=c.opos("box0"); H=np.array([0.1,0.15,0.1])
# grasp at +y top edge midpoint
G=np.array([B[0], B[1]+H[1], 0.2])
sd=np.array([B[0], B[1]+H[1]+0.45])
ang=np.arctan2(sd[1],sd[0])
print("box",np.round(B,3),"standoff",np.round(sd,3))
# drive: first go to a point on the ray from origin, then to standoff
print("dr1",c.gotobase(sd[0],sd[1],-np.pi/2),"b",np.round(c.robot()[0],3))
print("hi",c.move_to([G[0],G[1],0.45],yaw=YAW))
for z in [0.35,0.28,0.24,0.22,0.20]:
    r=c.move_to([G[0],G[1],z],yaw=YAW)
    if not r: print("blocked",z); break
c.grip(True); g=c.robot()[2]; print("grasp",g,"gpz",z)
if g<0.5: env.close(); sys.exit()
print("lift",c.move_to([G[0],G[1],0.55],yaw=YAW),"box",np.round(c.opos("box0"),3))
# drive to front of table
print("dr2",c.gotobase(0.0,0.0,0.0),"b",np.round(c.robot()[0],3))
tgt=np.array([0.6,0.15,0.6])
for p in [[G[0],G[1],0.6],[0.3,0.15,0.65],[0.5,0.15,0.62],[0.6,0.15,0.605],[0.6,0.15,0.6]]:
    r=c.move_to(p,yaw=YAW); print("mv",p,r,"box",np.round(c.opos("box0"),3))
c.grip(False)
print("rel grasp",c.robot()[2],"box",np.round(c.opos("box0"),3),"done",c.done)
for i in range(3):
    t,tr=c.step(np.zeros(11)); print("noop term",t,"trunc",tr)
print("steps",c.steps)
env.close()
