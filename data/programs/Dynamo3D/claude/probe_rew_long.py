from env_client import make_env
import numpy as np
env=make_env(); obs,info=env.reset(seed=11)
a=np.zeros(11,dtype=np.float32)
rs={}
i=0
while True:
    obs,r,te,tr,inf=env.step(a); i+=1
    rr=repr(float(r)); rs[rr]=rs.get(rr,0)+1
    if te or tr: print("end at",i,"te",te,"tr",tr,"r",r); break
    if i>2000: print("no end by 2000"); break
print(rs)
env.close()
