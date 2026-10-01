import sys, numpy as np
from env_client import make_env
import robot
from s2 import *
name=sys.argv[1]; hs=[float(v) for v in sys.argv[2].split(',')]
for h in hs:
    for lat in [0.0,0.01,0.02,0.03,0.04]:
        env=make_env(); obs,info=env.reset(seed=0)
        p=opos(obs,name)
        obs,bl,err=go_world(env,obs,p+np.array([lat,0,h]),robot.down_R(0.0),nmax=80)
        a=np.zeros(11,dtype=np.float32); a[10]=-1.0
        obs,r,t,tr,inf=env.step(a)
        ri=rinfo(obs)
        print(name,"h",round(h,3),"lat",lat,"blocked",bl,"err",round(err,4),"grasp",ri['grasp_active'],"tf",[ri['grasp_tf_x'],ri['grasp_tf_y'],ri['grasp_tf_z']],flush=True)
        env.close()
