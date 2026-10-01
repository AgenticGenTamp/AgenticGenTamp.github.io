from env_client import make_env
import numpy as np, time
env = make_env()
obs, info = env.reset(seed=0)
R = env.observation_space.get_type("robot")
rob = obs.get_objects(R)[0]
feats = env.observation_space.type_features[R]
def show(o):
    r=o.get_objects(R)[0]; print([round(float(o.get(r,f)),3) for f in feats[:11]])
t=time.time()
for i in range(4):
    a=np.zeros(11,dtype=np.float32); a[0]=0.2
    obs,rew,term,trunc,info=env.step(a); show(obs); print(rew,term,trunc,info)
print("time/step",(time.time()-t)/4)
for j in range(3,10):
    a=np.zeros(11,dtype=np.float32); a[j]=0.1
    obs,*_=env.step(a); show(obs)
a=np.zeros(11,dtype=np.float32); a[10]=-1
obs,*_=env.step(a); show(obs)
a=np.zeros(11,dtype=np.float32); a[2]=0.2
obs,*_=env.step(a); show(obs)
a=np.zeros(11,dtype=np.float32); a[0]=0.2
obs,*_=env.step(a); show(obs)
env.close()
