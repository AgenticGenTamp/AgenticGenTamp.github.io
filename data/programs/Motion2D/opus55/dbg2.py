from env_client import make_env
import numpy as np
env = make_env(); obs, info = env.reset(seed=159)
r=obs.get_object_from_name('robot')
P=lambda: tuple(round(float(obs.get(r,f)),4) for f in ['x','y','theta'])
def st(a):
    global obs
    obs,*_=env.step(np.array(a,dtype=np.float32))
for i in range(5): st([0,0,0.1748,0,0])
print(P())
for i in range(40): st([0,0.05,0,0,0])
print('after up', P())
for d in [0.02,0.01,0.005,0.002,0.001]:
    for i in range(20): st([0,d,0,0,0])
print('fine up', P())
for i in range(20): st([0.05,0,0,0,0])
print('right', P())
for th in [np.pi/2]:
    pass
for i in range(3): st([-0.05,0,0,0,0])
print(P())
for d in [0.0008,0.0004,0.0002,0.0001,0.00005,0.00002,0.00001]:
    for i in range(5): st([0,d,0,0,0])
print('fine up', P(), repr(float(obs.get(r,'y'))))
for d in [0.05,0.01,0.002]:
    for i in range(20): st([d,0,0,0,0])
print('right', P())
