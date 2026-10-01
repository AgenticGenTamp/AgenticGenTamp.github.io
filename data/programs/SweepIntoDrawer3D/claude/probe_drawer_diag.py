import numpy as np
from env_client import make_env
from ctrl2 import move_base, RFWD
from probe_drawer_lib import moveto_w, tip_world, w2l
np.set_printoptions(precision=3,suppress=True)
env=make_env(); obs,_=env.reset(seed=0)
print("base0",np.round(obs[125:128],3),"tip0",np.round(tip_world(obs),3))
obs=move_base(env,obs,[1.60,obs[126],obs[127]],steps=50,grip=0.0)
print("base",np.round(obs[125:128],3),"tip",np.round(tip_world(obs),3))
seq=[(1.25,-0.10,0.60),(1.15,-0.10,0.40),(1.05,-0.10,0.30),(0.95,-0.10,0.30),(0.75,-0.10,0.30)]
for p in seq:
    obs,d=moveto_w(env,obs,p,R=RFWD,steps=160,grip=0.0,multi=True)
    print("tgt",p,"-> tip",d['tip'],"err",round(d['err'],3),"ikres",round(d['ikres'],4),"clip",round(d['clip'],3),"used",d['used'],"dr",np.round(obs[103:109],3))
env.close()
