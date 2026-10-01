import numpy as np, sys, time
from env_client import make_env
from approach import *
from relaxed import solve_relaxed
env=make_env()
ap=GeneratedApproach(env.action_space, env.observation_space, {})
tm=float(sys.argv[3])
S1=[];S2=[]
for seed in range(int(sys.argv[1]),int(sys.argv[2])):
    obs,info=env.reset(seed=seed); x0=ap._x(obs)
    b1=9;b2=9; t=time.time()
    for n,o in ap._cubes(obs):
        p,cy=ap._cube_info(obs,o)
        for k in range(4):
            yaw=cy+k*np.pi/4
            sol=solve_relaxed(x0,p,yaw,tilt_max=tm)
            if sol: b2=min(b2,sol[1])
    S2.append(b2)
    print(seed, round(b2,3), round(time.time()-t,1), flush=True)
S2=np.array(S2); print('tilt',tm,'N1',np.mean(S2<=0.4),'N2',np.mean((S2>0.4)&(S2<=0.8)),'N3',np.mean(S2>0.8))
env.close()
