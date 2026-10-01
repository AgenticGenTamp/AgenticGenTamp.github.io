import sys, numpy as np
from env_client import make_env
def rob(obs):
    r=obs.get_object_from_name("robot"); f=obs.type_features[r.type]; d=obs.data[r]
    return {n:float(d[f.index(n)]) for n in f}
sign=float(sys.argv[1]); joints=[int(x) for x in sys.argv[2].split(',')]
env=make_env()
for j in joints:
    obs,_=env.reset(seed=0); key='pos_arm_joint%d'%(j+1)
    start=rob(obs)[key]
    a=np.zeros(11,np.float32); a[3+j]=0.1*sign
    prev=start; sat=None
    for i in range(200):
        obs,*_=env.step(a); q=rob(obs)[key]
        if sat is None and abs(q-prev)<1e-4 and i>3: sat=i+1
        prev=q
    print("joint%d start=%+.4f dir=%+g final=%+.4f satstep=%s"%(j+1,start,sign,prev,sat)); sys.stdout.flush()
env.close()
