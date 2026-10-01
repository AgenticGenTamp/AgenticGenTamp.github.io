import numpy as np
from env_client import make_env
from ctrl2 import move_base, RFWD
from probe_drawer_lib import moveto_w, tip_world
np.set_printoptions(precision=3,suppress=True)
env=make_env(); obs,_=env.reset(seed=0)
obs=move_base(env,obs,[1.60,obs[126],obs[127]],steps=50,grip=0.0)
tot=50
seq=[(1.20,-0.10,0.55),(1.00,-0.10,0.40),(0.90,-0.10,0.35)]
for p in seq:
    log=[]
    obs,d=moveto_w(env,obs,p,R=RFWD,steps=300,grip=0.0,multi=False,stall_n=40,log=log)
    tot+=d['used']
    print("tgt",p,"tip",d['tip'],"err",round(d['err'],3),"ikres",round(d['ikres'],4),"used",d['used'],"tot",tot)
    print("   q ",np.round(obs[128:135],3))
    print("   qd",np.round(d['qd'],3))
    print("   errtrace",[e for i,e in log[::20]])
env.close()
