import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True, linewidth=150)
env = make_env()
def run(seq, j=3):
    obs, info = env.reset(seed=0)
    out=[]
    for v in seq:
        a=np.zeros(11,dtype=np.float32); a[j]=v
        obs,*_=env.step(a); out.append(obs[125+j])
    print(seq); print(np.array(out))
run([0.1,0,0.1,0,0.1,0,0.1,0,0,0,0,0])
run([0.1,0,0,0,0.1,0,0,0,0.1,0,0,0,0,0])
run([0.05,0.05,0.05,0.05,0,0,0,0])
run([0.1]*20)
run([-0.1]*10+[0]*5)
env.close()
