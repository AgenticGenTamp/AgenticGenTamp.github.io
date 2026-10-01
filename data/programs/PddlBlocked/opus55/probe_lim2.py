from env_client import make_env
import numpy as np, sys
from envutil import Sim
env=make_env(); S=Sim(env,env.reset(seed=0)[0])
x=float(sys.argv[1]); yaw=float(sys.argv[2])
S.moveto(np.array([2.5,2.0,yaw]),S.q()); S.moveto(np.array([x,2.0,yaw]),S.q())
print('at',S.base())
for k in range(200):
    y0=S.base()[1]; S.step(np.r_[0,-0.01,np.zeros(9)])
    if abs(S.base()[1]-y0)<1e-6: break
print('x',x,'yaw',yaw,'blocked at y',S.base()[1])
