from env_client import make_env
import numpy as np, kin
from rutil import Bot
env=make_env(); obs,_=env.reset(seed=0); b=Bot(env,obs)
kin.set_params(mount_xyz=[0.12,0.0,0.3945],mount_yaw=0.0,tip_len=0.145)
b.goto(base_t=[1.1,1.2,-np.pi/2],grip=1.0)
bp=b.base(); tgt=np.array([1.1,0.7,0.05])
q,ok,e=kin.ik(bp,b.q(),tgt,'down',yaw=0.0); b.goto(q_t=q,tol=0.001)
print('at',np.round(b.tip()[0],4))
for z in np.arange(0.04,-0.04,-0.004):
    q,ok,e=kin.ik(bp,b.q(),[1.1,0.7,z],'down',yaw=0.0)
    b.goto(q_t=q,tol=0.0005,max_steps=15)
    print(round(z,3),'fk z',round(b.tip()[0][2],4),'qerr',round(np.abs(q-b.q()).max(),4))
