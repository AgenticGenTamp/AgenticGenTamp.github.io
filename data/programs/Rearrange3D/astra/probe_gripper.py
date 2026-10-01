from env_client import make_env
import numpy as np,json
np.set_printoptions(precision=4,suppress=True)
e=make_env();s,_=e.reset(seed=0)
for k in range(10):
 a=np.zeros(11);a[-1]=1
 s,r,t,tr,i=e.step(a)
 print(k,'robot',s[93:104],'objects',s[[0,1,2,16,17,18,32,33,34]],r)
json.dump(s.tolist(),open('closed_state.json','w'))
e.close()
