import numpy as np
from env_client import make_env
import robot
from s2 import *
env=make_env(); obs,info=env.reset(seed=0)
p=opos(obs,'obstruction0'); h=half(obs,'obstruction0')
gz = p[2]+h[2]+0.030
obs,bl,err=go_world(env,obs,np.array([p[0],p[1],0.28]),robot.down_R(0.0),nmax=60)
obs,bl,err=go_world(env,obs,np.array([p[0],p[1],gz]),robot.down_R(0.0),nmax=40)
a=np.zeros(11,dtype=np.float32); a[10]=-1.0; obs,r,t,tr,i=env.step(a)
print("grasp",rinfo(obs))
# lift and move
obs,bl,err=go_world(env,obs,np.array([p[0],p[1],0.28]),robot.down_R(0.0),nmax=40)
print("lifted obj",np.round(opos(obs,'obstruction0'),4),"blocked",bl)
obs,bl,err=go_world(env,obs,np.array([0.25,-0.1,0.28]),robot.down_R(0.0),nmax=60)
print("moved obj",np.round(opos(obs,'obstruction0'),4),"blocked",bl,"err",err)
# release in mid-air
a=np.zeros(11,dtype=np.float32); a[10]=1.0; obs,r,t,tr,i=env.step(a)
print("after open midair grasp",rinfo(obs)['grasp_active'],"obj",np.round(opos(obs,'obstruction0'),4))
# min gripper z while holding: re-grasp
obs,bl,err=go_world(env,obs,np.array([0.25,-0.1,opos(obs,'obstruction0')[2]+h[2]+0.030]),robot.down_R(0.0),nmax=40)
a=np.zeros(11,dtype=np.float32); a[10]=-1.0; obs,r,t,tr,i=env.step(a)
print("regrasp",rinfo(obs)['grasp_active'], rinfo(obs)['grasp_tf_z'])
zmin=None
for Z in np.arange(0.28,0.09,-0.005):
    obs,bl,err=go_world(env,obs,np.array([0.25,-0.1,Z]),robot.down_R(0.0),nmax=15)
    if bl: zmin=Z+0.005; break
print("min gripper z while holding:",round(zmin,3) if zmin else None,"obj",np.round(opos(obs,'obstruction0'),4))
a=np.zeros(11,dtype=np.float32); a[10]=1.0; obs,r,t,tr,i=env.step(a)
print("release at low:",rinfo(obs)['grasp_active'],"obj",np.round(opos(obs,'obstruction0'),4))
