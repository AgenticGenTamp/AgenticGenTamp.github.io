import numpy as np, sys
from env_client import make_env
np.set_printoptions(precision=3,suppress=True,linewidth=200)
env=make_env()
for s in map(int,sys.argv[1:]):
    o,_=env.reset(seed=s)
    print(s,'bowl',o[0:3],'drink',o[16:19],'q',o[19:23],'can',o[32:35],'base',o[93:96],'d',np.linalg.norm(o[16:18]-o[0:2]).round(3),np.linalg.norm(o[32:34]-o[0:2]).round(3))
