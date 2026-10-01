import numpy as np, sys
from env_client import make_env
np.set_printoptions(precision=4,suppress=True,linewidth=200)
seed=int(sys.argv[1])
env=make_env(); o,_=env.reset(seed=seed)
for t in range(int(sys.argv[2]) if len(sys.argv)>2 else 70):
    if t%int(sys.argv[3] if len(sys.argv)>3 else 5)==0: print(t, o[16:23], np.linalg.norm(o[23:26]).round(3))
    o,*_=env.step(np.zeros(11,np.float32))
