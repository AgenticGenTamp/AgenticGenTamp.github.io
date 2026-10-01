from env_client import make_env
import numpy as np
def rob(obs):
    r=obs.get_object_from_name("robot"); f=obs.type_features[r.type]; d=obs.data[r]
    return {n:float(d[f.index(n)]) for n in f}
env=make_env(); obs,_=env.reset(seed=0)
key='pos_arm_joint1'; a=np.zeros(11,np.float32); a[3]=0.1
p=rob(obs)[key]; ds=[]
for i in range(40):
    obs,*_=env.step(a); q=rob(obs)[key]; ds.append(q-p); p=q
print("joint1 per-step deltas (40):")
for i in range(0,40,5): print(" ",i,[round(v,5) for v in ds[i:i+5]])
print("mean steps30-40 =%.5f ratio=%.3f"%(np.mean(ds[30:]),np.mean(ds[30:])/0.1))
# release
a[3]=0.0; rel=[]
for i in range(10):
    obs,*_=env.step(a); q=rob(obs)[key]; rel.append(round(q-p,5)); p=q
print("after release:",rel)
env.close()
# per-joint steady ratio
env=make_env()
for j in range(7):
    obs,_=env.reset(seed=0); key='pos_arm_joint%d'%(j+1)
    a=np.zeros(11,np.float32); a[3+j]=0.1
    p=rob(obs)[key]; ds=[]
    for i in range(30):
        obs,*_=env.step(a); q=rob(obs)[key]; ds.append(q-p); p=q
    print("joint%d steady mean(last10)=%.5f ratio=%.3f"%(j+1,np.mean(ds[-10:]),np.mean(ds[-10:])/0.1))
env.close()
