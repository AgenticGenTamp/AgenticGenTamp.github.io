from env_client import make_env
from kin import *; from helpers import *
env=make_env(); obs,info=env.reset(seed=1)
s=rstate(obs); q=s[3:10]
Rd=np.array([[0,1,0],[1,0,0],[0,0,-1.]])
for z in [0.0,-0.1,-0.2,-0.25,-0.3,-0.33,-0.36,-0.39,-0.42]:
    qd,_,_=ik(np.array([0.5,0.3,z]),Rd,q)
    obs,t,ok=drive(env,obs,np.concatenate([s[:3],qd]),0.0,steps=150,tol=0.005)
    q=rstate(obs)[3:10]
    print(z,t,ok,'err',np.abs(wrap(qd-q)).max().round(4),'fk',fk_arm(q)[:3,3].round(3), cubes(obs)['cube1'][:3].round(3))
