import numpy as np, fk, sys
from env_client import make_env
from lib_util import robot, step_to
from expB_common import *
from expB_lib import *
def run(lat, ret, base_concurrent=True):
    env=make_env(); obs,s,g,b,d3,yaw=setup(env,1); R=fk.grasp_R(yaw); tot=0
    perp=np.array([-d3[1],d3[0],0.0])
    if base_concurrent:
        q0,_=fk.ik(b-d3*0.20,R,BASE,robot(obs)[3:10],seeds=6)
        obs,r,n=step_to(env,obs,BASE,q0); tot+=n
    else:
        obs,_,n=step_to(env,obs,BASE,robot(obs)[3:10]); tot+=n
        q0,_=fk.ik(b-d3*0.20,R,BASE,robot(obs)[3:10],seeds=6)
        obs,r,n=step_to(env,obs,BASE,q0); tot+=n
    qg,_=fk.ik(b-d3*0.03,R,BASE,q0,seeds=6)
    obs,r,n=step_to(env,obs,BASE,qg); tot+=n
    if r: print("approach blocker failed"); env.close(); return
    obs=grip(env,obs,-1.0); tot+=1
    ga=robot(obs)[11]
    p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
    obs,r1,n,p=cart(env,obs,R,p+perp*lat,step=0.02,verbose=False); tot+=n
    obs=grip(env,obs,1.0); tot+=1
    blk=blockpos(obs,'blocker')
    obs,r2,n,p=cart(env,obs,R,p-d3*ret,step=0.03,verbose=False); tot+=n
    q,_=fk.ik(g-d3*0.20,R,BASE,robot(obs)[3:10],seeds=6)
    obs,r3,n=step_to(env,obs,BASE,q); tot+=n
    obs,r4,n,p=cart(env,obs,R,g-d3*0.03,step=0.02,verbose=False); tot+=n
    before=blockpos(obs,'green0')
    obs=grip(env,obs,-1.0); tot+=1; ga2=robot(obs)[11]
    p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
    obs,r5,n5,p=cart(env,obs,R,p+np.array([0,0,0.06]),step=0.02,verbose=False)
    moved=float(np.linalg.norm(blockpos(obs,'green0')-before))
    print(f"lat={lat} ret={ret} conc={base_concurrent}: blkgrasp={ga} blk_final={np.round(blk,3).tolist()} "
          f"rejs={(r1,r2,r3,r4)} green0_ga={ga2} lifted={moved:.3f} STEPS_TO_HOLD={tot}")
    sys.stdout.flush(); env.close()
run(0.30,0.12,True)
run(0.20,0.12,True)
run(0.16,0.10,True)
run(0.12,0.10,True)
