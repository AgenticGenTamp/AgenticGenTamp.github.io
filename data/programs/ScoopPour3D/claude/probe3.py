from env_client import make_env
import numpy as np, time
env = make_env()
obs, info = env.reset(seed=1, options={'object_count':100})
r = obs.get_object_from_name('robot')
q = lambda o: o.data[r][3:10].copy()
t0=time.time()
for i in range(100):
    obs,_,_,_,_ = env.step(np.zeros(11,dtype=np.float32))
print("100 steps (100 cubes) wall time", round(time.time()-t0,2))
for amp in [0.01,0.02,0.05]:
    a=np.zeros(11,dtype=np.float32); a[3]=amp
    q0=q(obs)
    for i in range(20): obs,_,_,_,_=env.step(a)
    print('amp',amp,'delta/step',round((q(obs)[0]-q0[0])/20,4))
    a=np.zeros(11,dtype=np.float32)
    for i in range(10): obs,_,_,_,_=env.step(a)
    print('   after settle drift', round(q(obs)[0]-q0[0],4))
env.close()
