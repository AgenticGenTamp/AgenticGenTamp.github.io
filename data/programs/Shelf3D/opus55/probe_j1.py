from env_client import make_env
from helpers import *
from approach import Q_PICK0
env=make_env()
obs,info=env.reset(seed=1)
s=rstate(obs)
for tgt in [np.concatenate([[2.6],HOME[1:]]), Q_PICK0]:
    obs,t,ok=drive(env,obs,np.concatenate([s[:3],tgt]),0.0,steps=200,tol=0.01,gain=2.0)
    print(ok,t,rstate(obs)[3:10].round(2))
