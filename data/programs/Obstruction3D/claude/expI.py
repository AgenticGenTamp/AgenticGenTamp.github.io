import sys, numpy as np
from env_client import make_env
import robot
from s2 import *
gaps=[float(v) for v in sys.argv[1].split(',')]
for gap in gaps:
    env=make_env(); obs,info=env.reset(seed=1)
    p=opos(obs,'target_block'); h=half(obs,'target_block')
    reg=opos(obs,'target_region'); rh=half(obs,'target_region')
    gz=p[2]+h[2]+0.031
    obs,bl,err=go_world(env,obs,np.array([p[0],p[1],0.28]),robot.down_R(0.0),nmax=60)
    obs,bl,err=go_world(env,obs,np.array([p[0],p[1],gz]),robot.down_R(0.0),nmax=40)
    a=np.zeros(11,dtype=np.float32); a[10]=-1.0; obs,r,t,tr,i=env.step(a)
    if rinfo(obs)['grasp_active']<0.5: print("gap",gap,"GRASP FAIL"); continue
    G=grasp_tf(obs)
    obs,bl,err=go_world(env,obs,np.array([p[0],p[1],0.28]),robot.down_R(0.0),nmax=40)
    zc=reg[2]+rh[2]+h[2]+gap
    Tobj=np.eye(4); Tobj[:3,3]=[reg[0],reg[1],zc]
    Tg=Tobj@np.linalg.inv(G)
    obs,bl,err=go_world(env,obs,np.array([reg[0],reg[1],0.28]),robot.down_R(0.0),nmax=60)
    obs,bl,err=go_world(env,obs,Tg[:3,3],Tg[:3,:3],nmax=60,tol=1e-5)
    bp=opos(obs,'target_block')
    a=np.zeros(11,dtype=np.float32); a[10]=1.0
    obs,r,term,tr,i=env.step(a)
    print("gap",gap,"blocked",bl,"err",round(err,5),"blockz",round(bp[2],5),"want",round(reg[2]+rh[2]+h[2],5),"released",rinfo(obs)['grasp_active']<0.5,"TERM",term,flush=True)
    env.close()
