import sys, numpy as np
from env_client import make_env
def rob(obs):
    r=obs.get_object_from_name("robot"); f=obs.type_features[r.type]; d=obs.data[r]
    return {n:float(d[f.index(n)]) for n in f}
sign=float(sys.argv[1]); joints=[int(x) for x in sys.argv[2].split(',')]
env=make_env()
for j in joints:
    obs,_=env.reset(seed=0); key='pos_arm_joint%d'%(j+1)
    a=np.zeros(11,np.float32); a[3+j]=0.1*sign
    prev=rob(obs)[key]; stall=None; stallval=None
    for i in range(320):
        obs,*_=env.step(a); q=rob(obs)[key]
        if stall is None and abs(q-prev)<2e-3 and i>2: stall=i+1; stallval=q
        prev=q
    print("joint%d dir=%+g stall@%s val=%s final=%+.4f"%(j+1,sign,stall,None if stallval is None else round(stallval,4),prev)); sys.stdout.flush()
env.close()
