import numpy as np, fk, sys
from env_client import make_env
from lib_util import robot, step_to
from expB_common import *
from expB_lib import *
env=make_env(); obs,s,g,b,d3,yaw=setup(env,1); R=fk.grasp_R(yaw); tot=0
perp=np.array([-d3[1],d3[0],0.0])
obs,_,n=step_to(env,obs,BASE,robot(obs)[3:10]); tot+=n; print("1 base",n)
q,_=fk.ik(b-d3*0.20,R,BASE,robot(obs)[3:10],seeds=6)
obs,r,n=step_to(env,obs,BASE,q); tot+=n; print("2 pregrasp blocker",n,"rej",r)
q,_=fk.ik(b-d3*0.03,R,BASE,q,seeds=6)
obs,r,n=step_to(env,obs,BASE,q); tot+=n; print("3 grasp pose",n,"rej",r)
obs=grip(env,obs,-1.0); tot+=1; print("4 close ga",robot(obs)[11])
p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
obs,r,n,p=cart(env,obs,R,p+perp*0.30,tag="5 lat0.30",step=0.02); tot+=n
obs=grip(env,obs,1.0); tot+=1; print("6 open ga",robot(obs)[11],"blk",np.round(blockpos(obs,'blocker'),3).tolist())
obs,r,n,p=cart(env,obs,R,p-d3*0.12,tag="7 retract",step=0.03); tot+=n
q,_=fk.ik(g-d3*0.20,R,BASE,robot(obs)[3:10],seeds=6)
obs,r,n=step_to(env,obs,BASE,q); tot+=n; print("8 pregrasp green0",n,"rej",r)
obs,r,n,p=cart(env,obs,R,g-d3*0.03,tag="9 approach green0",step=0.02); tot+=n
print("9 rej",r,"tool",np.round(p,3).tolist(),"dist",round(float(np.dot(g-p,d3)),3))
before=blockpos(obs,'green0')
obs=grip(env,obs,-1.0); tot+=1
print("10 close ga",robot(obs)[11],"TOTAL",tot)
p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
obs,r,n,p=cart(env,obs,R,p+np.array([0,0,0.06]),tag="11 verify-lift",step=0.02); tot+=n
print("green0 now",np.round(blockpos(obs,'green0'),3).tolist(),"moved",round(float(np.linalg.norm(blockpos(obs,'green0')-before)),3),
      "blocker",np.round(blockpos(obs,'blocker'),3).tolist(),"ga",robot(obs)[11],"TOTAL_incl_verify",tot)
env.close()
