from env_client import make_env
import numpy as np,json
np.set_printoptions(precision=4,suppress=True)
e=make_env();s,_=e.reset(seed=0)
for k in range(30):
 a=np.zeros(11);a[10]=1
 s,r,t,tr,i=e.step(a)
print('OPEN',s[16:38]);json.dump(s.tolist(),open('open_state.json','w'));e.close()
