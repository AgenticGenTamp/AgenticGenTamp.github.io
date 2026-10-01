from env_client import make_env
import numpy as np

env = make_env()
o, _ = env.reset(seed=0)
print('initial objects', np.round(o[:48], 4))
print('robot', np.round(o[93:115], 4))
for dim in range(11):
    before = o.copy()
    a = np.zeros(11, np.float32)
    a[dim] = 0.1 if dim < 10 else 1.0
    # gripper command otherwise retain likely zero/closed
    for k in range(5):
        o, r, t, tr, inf = env.step(a)
    print('dim', dim, 'r', r, 'objdelta', np.round(o[[0,1,2,16,17,18,32,33,34]]-before[[0,1,2,16,17,18,32,33,34]],4), 'robotdelta', np.round(o[93:104]-before[93:104],4))
env.close()
