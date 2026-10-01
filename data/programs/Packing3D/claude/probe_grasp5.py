import numpy as np
from probe_lib import *
env=make_env(); obs,info=env.reset(seed=0)
base=rb(obs); p=ppos(obs,'part0')
obs=grip(env,obs,1.0)
obs,_,_=goto(env,obs,(p[0]-0.10,p[1],0.35),Rdown,base)
obs,_,m=goto(env,obs,(p[0]-0.10,p[1],0.30),Rdown,base)
obs=grip(env,obs,-1.0,n=1)
print("grasp after 1 close step",rfeat(obs,'grasp_active'),"finger",rfeat(obs,'finger_state'))
obs,_,_=goto(env,obs,(p[0]-0.10,p[1],0.40),Rdown,base)
print("lifted part",np.round(ppos(obs,'part0'),3))
obs=grip(env,obs,1.0,n=1)
print("open1 grasp",rfeat(obs,'grasp_active'),"finger",rfeat(obs,'finger_state'),"part",np.round(ppos(obs,'part0'),3))
obs,r,t,tr,i=env.step(np.zeros(11))
print("idle    grasp",rfeat(obs,'grasp_active'),"part",np.round(ppos(obs,'part0'),3))
obs=grip(env,obs,1.0,n=3)
print("open3 grasp",rfeat(obs,'grasp_active'),"part",np.round(ppos(obs,'part0'),3))
# move arm to see if part still attached
obs,_,_=goto(env,obs,(p[0]-0.05,p[1],0.40),Rdown,base)
print("moved: fk",np.round(fkpos(obs),3),"part",np.round(ppos(obs,'part0'),3),"grasp",rfeat(obs,'grasp_active'))
# try lowering until near table then open
for z in [0.36,0.33,0.31,0.30,0.29]:
    obs,blk,m=goto(env,obs,(p[0]-0.05,p[1],z),Rdown,base)
    print("z",z,m,"part",np.round(ppos(obs,'part0'),3))
    if blk: break
obs=grip(env,obs,1.0,n=2)
print("open near table: grasp",rfeat(obs,'grasp_active'),"part",np.round(ppos(obs,'part0'),3),"finger",rfeat(obs,'finger_state'))
obs,_,_=goto(env,obs,(p[0]-0.05,p[1],0.40),Rdown,base)
print("retreat: fk",np.round(fkpos(obs),3),"part",np.round(ppos(obs,'part0'),3),"grasp",rfeat(obs,'grasp_active'))
env.close()
