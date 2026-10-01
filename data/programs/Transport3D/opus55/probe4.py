from env_client import make_env
import numpy as np
from kin import fk_arm
env = make_env()
J=['joint_%d'%i for i in range(1,8)]
def q(obs):
    r=obs.get_object_from_name('robot'); return [float(obs.get(r,f)) for f in J]
obs,_=env.reset(seed=1)
for k in range(60):
    a=np.zeros(11,dtype=np.float32); a[4]=0.05
    obs,*_=env.step(a)
    qq=q(obs); M=fk_arm(qq)
    print(k, round(qq[1],3), np.round(M[:3,3],3), "tip z(+0.12):", round((M[:3,3]+0.12*M[:3,2])[2],3))
    if k>0 and abs(qq[1]-prev)<1e-6: break
    prev=qq[1]
env.close()
