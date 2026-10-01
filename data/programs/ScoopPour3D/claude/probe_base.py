from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=1)
r = obs.get_object_from_name('robot')
def b(o): return np.round(o.data[r][:3],3)
print('init', b(obs))
for name, vec in [('y+',[0,0.1,0]), ('x+',[0.1,0,0]), ('th+',[0,0,0.1])]:
    a=np.zeros(11,dtype=np.float32); a[:3]=vec
    for i in range(15):
        obs,rew,term,trunc,info = env.step(a)
    print(name, b(obs))
    a=np.zeros(11,dtype=np.float32)
    for i in range(5): obs,_,_,_,_=env.step(a)
    print(name,'settled', b(obs))
env.close()
