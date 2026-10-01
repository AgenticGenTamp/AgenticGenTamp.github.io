import numpy as np,sys
import ctl, kinova_fk as K
from ctl import *
from env_client import make_env
ctl.Z_OFF=0.44
relx=float(sys.argv[1]); relz=float(sys.argv[2]) if len(sys.argv)>2 else 0.52
env=make_env(); obs,info=env.reset(seed=0)
o=np.asarray(obs); bowl=o[0:3].copy()
obs=base_goto(env,obs,o[93],o[94],0.0,steps=15)
obs=base_goto(env,obs,np.asarray(obs)[93],bowl[1]-0.40,0.0,steps=30)
obs=base_goto(env,obs,-0.20,bowl[1]-0.40,0.0,steps=12,vmax=0.05)
o=np.asarray(obs); bx=o[93]
# arm config: model TCP straight ahead at relx, height relz(world)
tgt=np.array([bx+relx, o[94], relz])
obs,u=servo(env,obs,tgt+np.array([0,0,0.1]),grip=0.0,max_steps=70)
obs,u=servo(env,obs,tgt,grip=0.0,max_steps=30,chunk=5)
qhold=np.asarray(obs)[96:103].copy()
o=np.asarray(obs)
print("relx",relx,"base",np.round(o[93:96],3),"modelTCP",np.round(ctl.tcp_world(obs),3),"bowl",np.round(o[0:3],3))
ref=np.asarray(obs)[0:3].copy(); prev=ref.copy()
for i in range(40):
    a=np.zeros(11,dtype=np.float32)
    q=np.asarray(obs)[96:103]
    a[3:10]=np.clip(3.0*(qhold-q),-0.1,0.1)
    a[1]=0.023
    obs,r,te,tr,inf=env.step(a)
    o=np.asarray(obs)
    d=np.linalg.norm(o[0:3]-prev)
    if d>0.003:
        print("CONTACT base_y=%.3f modelTCP=%s bowl=%s->%s"%(o[94],np.round(ctl.tcp_world(obs),3),np.round(prev,3),np.round(o[0:3],3)))
        break
    prev=o[0:3].copy()
else:
    print("NO CONTACT final base_y=%.3f modelTCP=%s"%(o[94],np.round(ctl.tcp_world(obs),3)))
env.close()
