import numpy as np
from env_client import make_env
J=["joint_%d"%i for i in range(1,8)]
env=make_env(); obs,info=env.reset(seed=0)
r=obs.get_object_from_name("robot")
feats=["finger_state","grasp_active","grasp_tf_x","grasp_tf_y","grasp_tf_z","grasp_tf_qw"]
def show(o,tag):
    r=o.get_object_from_name("robot")
    print(tag,[round(float(o.get(r,f)),3) for f in feats], "q",[round(float(o.get(r,f)),3) for f in J])
show(obs,"init")
for v,tag in [(1.0,"open"),(1.0,"open2"),(-1.0,"close"),(-1.0,"close2"),(0.6,"open3"),(-0.6,"close3")]:
    a=np.zeros(11); a[10]=v
    obs,rew,t,tr,i=env.step(a); show(obs,tag)
# move joint then close
a=np.zeros(11); a[3]=0.2; obs,*_=env.step(a); show(obs,"j1")
a=np.zeros(11); a[10]=-1.0; a[3]=0.1; obs,*_=env.step(a); show(obs,"move+close")
env.close()
