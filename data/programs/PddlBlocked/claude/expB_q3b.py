import numpy as np, fk, sys
from env_client import make_env
from lib_util import robot, step_to
from expB_common import *
base=np.array([3.72,0.10,0.0])
env=make_env()
obs,s,g,b,d3,yaw=setup(env,1)
R=fk.grasp_R(yaw)
obs,_,n0=step_to(env,obs,base,robot(obs)[3:10])
q_pre,_=fk.ik(b-d3*0.20,R,base,robot(obs)[3:10],seeds=6)
obs,_,n1=step_to(env,obs,base,q_pre)
q_g,_=fk.ik(b-d3*0.03,R,base,q_pre,seeds=6)
obs,_,n2=step_to(env,obs,base,q_g)
obs=grip(env,obs,-1.0)
print("grasped",robot(obs)[11],"steps",n0+n1+n2+1)


p0,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
perp=np.array([-d3[1],d3[0],0.0])
q0=robot(obs)[3:10].copy()
tests={"back0.02":-d3*0.02,"back0.05":-d3*0.05,"back0.10":-d3*0.10,
       "up0.05":np.array([0,0,0.05]),"up0.10":np.array([0,0,0.10]),
       "lat+0.05":perp*0.05,"lat-0.05":-perp*0.05,"fwd0.02":d3*0.02}
import copy
for name,dv in tests.items():
    # reset arm to grasp config each time by re-running? instead just try and undo
    qn,e=fk.ik(p0+dv,R,base,q0,seeds=1,w_rot=1.0,q_ref=q0,w_ref=0.05)
    obs,rej,n=step_to(env,obs,base,qn,grip=0.0,maxsteps=20)
    p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
    print(f"{name:10s} ikerr={e:.4f} rej={rej} n={n} moved={np.linalg.norm(p-p0):.4f} ga={robot(obs)[11]} blk={np.round(blockpos(obs,'blocker')[:2],3).tolist()}")
    # return
    obs,rej2,n2=step_to(env,obs,base,q0,grip=0.0,maxsteps=20)
    sys.stdout.flush()
env.close()
