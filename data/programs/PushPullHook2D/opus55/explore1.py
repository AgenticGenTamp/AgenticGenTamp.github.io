import numpy as np
from env_client import make_env
env = make_env()
np.set_printoptions(precision=3, suppress=True, linewidth=200)
obs, info = env.reset(seed=0)
def st(a):
    global obs
    obs, r, te, tr, _ = env.step(np.array(a, dtype=np.float32))
    return r, te
print(obs[:9])
for i in range(3):
    st([0.05,0,0,0,0]); print('dx', obs[:9])
for i in range(3):
    st([0,0.05,0,0,0]); print('dy', obs[:9])
for i in range(3):
    st([0,0,0.19,0,0]); print('dth', obs[:9])
for i in range(3):
    st([0,0,0,0.1,0]); print('darm+', obs[:9])
for i in range(3):
    st([0,0,0,-0.1,0]); print('darm-', obs[:9])
# go up to find the wall
for i in range(30):
    st([0,0.05,0,0,0])
print('after up', obs[:3])
for i in range(80):
    st([0.05,0,0,0,0])
print('after right', obs[:3])
for i in range(100):
    st([-0.05,0,0,0,0])
print('after left', obs[:3])
for i in range(40):
    st([0,-0.05,0,0,0])
print('after down', obs[:3])
