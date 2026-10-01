import numpy as np
from probe_lib import *
env=make_env(); obs,info=env.reset(seed=0)
base=rb(obs); p=ppos(obs,'part0')
print("part0",p)
obs=grip(env,obs,1.0)
obs,blk,msg=goto(env,obs,(p[0]-0.10,p[1],0.35),Rdown,base)
obs,blk,msg=goto(env,obs,(p[0]-0.10,p[1],0.30),Rdown,base)
print("approach",msg,fkpos(obs))
obs=grip(env,obs,-1.0)
print("grasp",rfeat(obs,'grasp_active'),"partga",pfeat(obs,'part0','grasp_active'),"finger",rfeat(obs,'finger_state'))
print("grasp_tf",[round(rfeat(obs,'grasp_tf_'+k),4) for k in ['x','y','z','qx','qy','qz','qw']])
print("part after close",ppos(obs,'part0'))
# lift
obs,blk,msg=goto(env,obs,(p[0]-0.10,p[1],0.38),Rdown,base)
print("lift",msg,"fk",np.round(fkpos(obs),3),"part",np.round(ppos(obs,'part0'),3))
# move sideways
obs,blk,msg=goto(env,obs,(0.30,0.0,0.38),Rdown,base)
print("move",msg,"fk",np.round(fkpos(obs),3),"part",np.round(ppos(obs,'part0'),3))
# rotate yaw while holding
obs,blk,msg=goto(env,obs,(0.30,0.0,0.38),Rdown@rotz(np.pi/4),base)
print("yaw45",msg,"fk",np.round(fkpos(obs),3),"part",np.round(ppos(obs,'part0'),3),
      "q",[round(pfeat(obs,'part0',f),3) for f in ['pose_qx','pose_qy','pose_qz','pose_qw']])
obs,blk,msg=goto(env,obs,(0.30,0.0,0.38),Rdown,base)
# descend toward rack
for z in [0.34,0.32,0.30,0.28,0.26]:
    obs,blk,msg=goto(env,obs,(0.30,0.0,z),Rdown,base)
    print("descend z",z,msg,"fk",np.round(fkpos(obs),3),"part",np.round(ppos(obs,'part0'),3))
    if blk: break
obs=grip(env,obs,1.0)
print("after open: grasp",rfeat(obs,'grasp_active'),"part",np.round(ppos(obs,'part0'),3),"finger",rfeat(obs,'finger_state'))
obs,r,t,tr,i=env.step(np.zeros(11))
print("after idle step: part",np.round(ppos(obs,'part0'),3),"term",t)
env.close()
