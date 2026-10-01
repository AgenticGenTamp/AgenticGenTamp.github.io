from env_client import make_env
import numpy as np
import sys
d=eval(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=11)
R=obs.get_object_from_name("robot"); C=obs.get_object_from_name("obstacle_chair")
rs={}
for i in range(500):
    a=np.zeros(11,dtype=np.float32); a[0]=0.1*d[0]; a[1]=0.1*d[1]
    obs,r,te,tr,inf=env.step(a)
    rr=repr(float(r)); rs[rr]=rs.get(rr,0)+1
    if te or tr:
        print(d,"TERM at",i,te,tr,"base",round(obs.get(R,"pos_base_x"),2),round(obs.get(R,"pos_base_y"),2)); break
print(d,"base",round(obs.get(R,"pos_base_x"),2),round(obs.get(R,"pos_base_y"),2),"rewards",rs)
env.close()
