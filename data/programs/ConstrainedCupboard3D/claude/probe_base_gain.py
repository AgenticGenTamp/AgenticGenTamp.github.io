from env_client import make_env
import numpy as np
def rob(obs):
    r=obs.get_object_from_name("robot"); f=obs.type_features[r.type]; d=obs.data[r]
    return {n:float(d[f.index(n)]) for n in f}
for dim,name in [(0,'base_x'),(2,'base_rot'),(3,'arm_joint1')]:
    key='pos_'+name
    for c in [0.01,0.025,0.05,0.075,0.1,-0.1]:
        env=make_env(); obs,_=env.reset(seed=0)
        a=np.zeros(11,np.float32); a[dim]=c
        p=rob(obs)[key]; ds=[]
        for i in range(8):
            obs,*_=env.step(a); q=rob(obs)[key]; ds.append(q-p); p=q
        print("%s cmd=%+.3f steady=%+.5f ratio=%.3f  first3=%s"%(name,c,ds[-1],ds[-1]/c,[round(v,5) for v in ds[:3]]))
        env.close()
