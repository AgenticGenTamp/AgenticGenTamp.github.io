import numpy as np
from env_client import make_env
from ik import ik
np.set_printoptions(precision=4,suppress=True,linewidth=200)
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)

def goto(env,obs,p,steps=25,R=Rdown):
    q=obs[128:135].copy()
    qd,_=ik(np.array(p),R,q)
    for i in range(steps):
        a=np.zeros(11); a[3:10]=np.clip((qd-obs[128:135]),-0.1,0.1)
        obs,r,t,tr,_=env.step(a)
    return obs,qd

env=make_env(); obs,_=env.reset(seed=0)
c0=obs[:80].reshape(5,16)[:,:3].copy()
# approach from above at x=0.75 then sweep toward robot
obs,_=goto(env,obs,[0.75,0.0,0.30],40)
obs,qd=goto(env,obs,[0.75,0.0,0.165],30)
print("start qerr",np.round(np.abs(qd-obs[128:135]).max(),3))
for x in np.arange(0.72,0.38,-0.02):
    obs,qd=goto(env,obs,[x,0.0,0.165],12)
    c=obs[:80].reshape(5,16)[:,:3]
    d=np.linalg.norm(c-c0,axis=1)
    print(f"x={x:.2f} qerr={np.abs(qd-obs[128:135]).max():.3f} cubemoved={np.round(d,3)}")
env.close()
