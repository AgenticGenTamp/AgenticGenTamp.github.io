"""Test local final-joint adjustments of the known box-grasp route on a cube."""
import math, sys
import numpy as np
from env_client import make_env
from probe_grasp_structured import command, val

target = sys.argv[1] if len(sys.argv) > 1 else "cube0"
dq2 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.
dq4 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.
dq6 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.
wp26=np.array([4.1558211745,.4267110066,-1.5021092155,-1.9006263446,
               1.8840013425,.0609092902,5.2361419160])
wp27=np.array([3.8707797770,1.7030209485,-2.8556712580,-.6279298538,
               4.2697360597,1.3194771600,7.2832584340])
wp27[1]+=dq2; wp27[3]+=dq4
wp27[5]+=dq6
env=make_env();state,_=env.reset(seed=1)
tx=val(state,target,"pose_x");ty=val(state,target,"pose_y")
routes=[(wp26,[.126212298,-.083534270,-3.139703362]),
        (wp27,[-.428185141,.054943428,2.978131848])]
for q,rel in routes:
    base=[tx+rel[0],ty+rel[1],rel[2]]
    state=command(env,state,base,q,1.,repeats=35)
    state=command(env,state,base,q,-1.,repeats=2)
if val(state,"robot","grasp_active")>.5:
    print("HIT",target,dq2,dq4,dq6,[val(state,"robot","joint_%d"%i) for i in range(1,8)])
else: print("MISS",target,dq2,dq4,dq6)
env.close()
