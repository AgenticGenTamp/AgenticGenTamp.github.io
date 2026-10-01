import numpy as np
from env_client import make_env

def rob(s):
    return s.get_object_from_name('robot')
def feat(s):
    r=rob(s)
    return np.array([s.get(r,'pos_arm_joint'+str(i)) for i in range(1,8)]+[s.get(r,'pos_gripper')])
for label,jdelta,jvel,g in [('hold',0,0,0),('delta',.1,0,1),('vel',0,1,0),('both',.1,1,1),('close',0,0,1)]:
    env=make_env(); s,info=env.reset(seed=0); initial=feat(s)
    print(label,'initial',np.round(initial,4).tolist(), 'info',info,flush=True)
    for t in range(10):
        a=np.zeros(18,np.float32);a[3]=jdelta;a[11]=jvel;a[10]=g
        s,r,done,tr,info=env.step(a)
        if t in [0,1,4,9]:print(label,t+1,'diff',np.round(feat(s)-initial,4).tolist(),'r',r,'done',done,flush=True)
    env.close()
