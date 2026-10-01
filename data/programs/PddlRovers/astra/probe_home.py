from env_client import make_env
import numpy as np
for x,y in [(.2,0),(.2,.1),(.2,.15),(.2,.19),(.2,.2)]:
 e=make_env();s,_=e.reset(seed=0);a=np.zeros(8);a[0]=x;a[1]=y;s,*_=e.step(a);r=s.get_object_from_name('rover0');print(x,y,s.get(r,'at_home'));e.close()
