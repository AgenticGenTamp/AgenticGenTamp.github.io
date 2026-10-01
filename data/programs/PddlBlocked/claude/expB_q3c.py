import numpy as np, fk, sys
from env_client import make_env
from lib_util import robot, step_to
from expB_common import *
base=np.array([3.72,0.10,0.0])

def grasp_blocker(env):
    obs,s,g,b,d3,yaw=setup(env,1); R=fk.grasp_R(yaw); tot=0
    obs,_,n=step_to(env,obs,base,robot(obs)[3:10]); tot+=n
    q,_=fk.ik(b-d3*0.20,R,base,robot(obs)[3:10],seeds=6)
    obs,_,n=step_to(env,obs,base,q); tot+=n
    q,_=fk.ik(b-d3*0.03,R,base,q,seeds=6)
    obs,_,n=step_to(env,obs,base,q); tot+=n
    obs=grip(env,obs,-1.0); tot+=1
    return obs,g,b,d3,yaw,R,tot

for lift in [0.0,0.05,0.10,0.15]:
    env=make_env()
    obs,g,b,d3,yaw,R,tot=grasp_blocker(env)
    perp=np.array([-d3[1],d3[0],0.0])
    p0,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
    ok=True
    if lift>0:
        obs,rej,n,p=move_cart(env,obs,R,p0+np.array([0,0,lift]),base,grip=0.0); tot+=n
        print(f"lift={lift}: rej={rej} n={n} tool={np.round(p,3).tolist()}")
        if rej: ok=False
    if ok:
        p1,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
        obs,rej,n,p=move_cart(env,obs,R,p1-d3*0.25,base,grip=0.0); tot+=n
        print(f"  retract0.25: rej={rej} n={n} tool={np.round(p,3).tolist()} ga={robot(obs)[11]} tot={tot}")
    sys.stdout.flush(); env.close()
