from env_client import make_env
import numpy as np, time
env = make_env()
obs,info = env.reset(seed=1)
r = obs.get_object_from_name("robot")
def rv(o): return np.round(o.data[r],4).tolist()
print("init", rv(obs))
t0=time.time()
# test joint 1 delta
a = np.zeros(11, dtype=np.float32); a[3]=0.2
obs,_,_,_,_ = env.step(a); print("j1+0.2", rv(obs))
a = np.zeros(11); a[0]=0.2; a[1]=0.1
obs,_,_,_,_ = env.step(a); print("base +0.2x +0.1y", rv(obs))
a = np.zeros(11); a[2]=0.2
obs,_,_,_,_ = env.step(a); print("rot+0.2", rv(obs))
a = np.zeros(11); a[0]=0.2
obs,_,_,_,_ = env.step(a); print("after rot, base x+0.2", rv(obs))
a = np.zeros(11); a[10]=1.0
obs,_,_,_,_ = env.step(a); print("open", rv(obs))
a = np.zeros(11); a[10]=-1.0
obs,_,_,_,_ = env.step(a); print("close", rv(obs))
print("elapsed for 6 steps", time.time()-t0)
env.close()
