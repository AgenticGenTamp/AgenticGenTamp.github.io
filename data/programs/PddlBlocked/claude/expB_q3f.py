import numpy as np, fk, sys
from env_client import make_env
from lib_util import robot, step_to
from expB_common import *
from expB_lib import *
def prep(env):
    obs,s,g,b,d3,yaw=setup(env,1); R=fk.grasp_R(yaw)
    obs,_,n=step_to(env,obs,BASE,robot(obs)[3:10])
    obs,t,rr=grasp_at(env,obs,R,b-d3*0.03)
    return obs,g,b,d3,R,n+t
for variant in ["nolift_lat0.30","lift_retract0.35","nolift_retract0.35"]:
    env=make_env(); obs,g,b,d3,R,tot=prep(env)
    perp=np.array([-d3[1],d3[0],0.0])
    print("==",variant,"ga",robot(obs)[11],"steps",tot)
    p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
    if variant=="nolift_lat0.30":
        obs,r,n,p=cart(env,obs,R,p+perp*0.30,tag="lat"); tot+=n
    elif variant=="lift_retract0.35":
        obs,r,n,p=cart(env,obs,R,p+np.array([0,0,0.05]),tag="lift"); tot+=n
        obs,r,n,p=cart(env,obs,R,p-d3*0.35,tag="retract"); tot+=n
    else:
        obs,r,n,p=cart(env,obs,R,p-d3*0.35,tag="retract"); tot+=n
    obs=grip(env,obs,1.0); tot+=1
    print(f"   open: ga={robot(obs)[11]} REFUSED={robot(obs)[11]>0.5} blk={np.round(blockpos(obs,'blocker'),3).tolist()} total={tot}")
    sys.stdout.flush(); env.close()
