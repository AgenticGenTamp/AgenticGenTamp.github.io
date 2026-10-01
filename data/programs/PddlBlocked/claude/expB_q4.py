import numpy as np, fk, sys
from env_client import make_env
from lib_util import robot, step_to
from expB_common import *
from expB_lib import *
for lat,cstep in [(0.30,0.02),(0.30,0.05),(0.20,0.05),(0.15,0.05)]:
    env=make_env(); obs,s,g,b,d3,yaw=setup(env,1); R=fk.grasp_R(yaw); tot=0
    obs,_,n=step_to(env,obs,BASE,robot(obs)[3:10]); tot+=n
    obs,t,rr=grasp_at(env,obs,R,b-d3*0.03); tot+=t
    ga1=robot(obs)[11]
    perp=np.array([-d3[1],d3[0],0.0])
    p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
    obs,r,n,p=cart(env,obs,R,p+perp*lat,tag=f"lat{lat}",step=cstep,verbose=False); tot+=n
    obs=grip(env,obs,1.0); tot+=1
    released=robot(obs)[11]<0.5
    blkpos=blockpos(obs,'blocker')
    # back off then approach green0
    obs,r2,n,p=cart(env,obs,R,g-d3*0.20,tag="to_pre",step=0.05,verbose=False); tot+=n
    q,e=fk.ik(g-d3*0.03,R,BASE,robot(obs)[3:10],seeds=6)
    obs,rej,n=step_to(env,obs,BASE,q); tot+=n
    p2,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
    obs=grip(env,obs,-1.0); tot+=1
    ga2=robot(obs)[11]
    gp=blockpos(obs,'green0')
    print(f"lat={lat} cstep={cstep}: grasp_blocker={ga1} released={released} blk={np.round(blkpos,3).tolist()} "
          f"pre_rej={r2} final_rej={rej} tool={np.round(p2,3).tolist()} GREEN0_GRASPED={ga2} TOTAL_STEPS={tot}")
    sys.stdout.flush(); env.close()
