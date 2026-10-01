import numpy as np, fk
from env_client import make_env
from lib_util import robot, step_to
from expB_common import *

env=make_env()
obs,s,g,b,d3,yaw=setup(env,1)
print("objects:",names(obs))
for k,v in s.items():
    if k!="robot": print(" ",k,np.round(v,4).tolist())
print("green0",np.round(g,4),"blocker",np.round(b,4),"dir",np.round(d3,4),"yaw",round(yaw,4))
R=fk.grasp_R(yaw)
base=np.array([3.72,0.10,0.0])
total=0
obs,rej,n=step_to(env,obs,base,robot(obs)[3:10]); total+=n
print("base move rejected",rej,"steps",n,"base",np.round(robot(obs)[:3],3))

def tool_target(dist):  # dist along -dir from green0
    return g-d3*dist

pre=tool_target(0.25); tgt=tool_target(0.03)
q_pre,e1=fk.ik(pre,R,base,robot(obs)[3:10],seeds=6)
q_g,e2=fk.ik(tgt,R,base,q_pre,seeds=6)
print("ik err pre",round(e1,4),"grasp",round(e2,4))
obs,rej1,n=step_to(env,obs,base,q_pre); total+=n
p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
print("pregrasp rejected",rej1,"steps",n,"tool",np.round(p,4),"dist_along",round(float(np.dot(g-p,d3)),4))
obs,rej2,n=step_to(env,obs,base,q_g); total+=n
p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
print("grasp move rejected",rej2,"steps",n,"tool",np.round(p,4),"dist_along",round(float(np.dot(g-p,d3)),4))
before={k:blockpos(obs,k) for k in names(obs) if k!="robot"}
obs=grip(env,obs,-1.0); total+=1
print("grasp_active",robot(obs)[11],"gripper_opening",round(float(robot(obs)[10]),4))
for k in before:
    now=blockpos(obs,k); dd=np.linalg.norm(now-before[k])
    if dd>1e-4: print("  moved:",k,round(float(dd),4))
print("TOTAL STEPS",total)
env.close()
