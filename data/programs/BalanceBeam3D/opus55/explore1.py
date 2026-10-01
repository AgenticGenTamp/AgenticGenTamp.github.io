from env_client import make_env
import numpy as np, time
np.set_printoptions(precision=4, suppress=True, linewidth=200)
env = make_env()
print(env.max_steps)
obs, info = env.reset(seed=0)
a = np.zeros(11, dtype=np.float32)
t=time.time()
for k in range(3):
    obs2, r, te, tr, info = env.step(a)
    print(r, te, tr, info, np.abs(obs2-obs).max())
print("time/step", (time.time()-t)/3)
# base move
a[0]=0.1
for k in range(5):
    obs2, r, te, tr, info = env.step(a)
    print("base", obs2[16:19], obs2[27:30])
a[:]=0; a[3]=0.1
for k in range(3):
    obs2, r, te, tr, info = env.step(a)
    print("j1", obs2[19:26])
a[:]=0; a[10]=1
for k in range(5):
    obs2, r, te, tr, info = env.step(a)
    print("grip", obs2[26], obs2[37])
env.close()
