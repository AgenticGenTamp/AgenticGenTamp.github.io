import numpy as np
from env_client import make_env
env=make_env(); obs,_=env.reset(seed=0); obs=np.asarray(obs,float)
b=obs.copy()
for c in range(5): b[c*16+0]=20; b[c*16+1]=20; b[c*16+2]=20
b[147]=20;b[148]=20;b[149]=20
b[125]=4.0; b[126]=4.0
print('ref', env.render_state(state=b.tolist(), label="cl_ref"))
for j in range(6):
    for v,tag in [(0.45,'a'),(0.65,'b')]:
        o=b.copy(); o[103+j]=v
        print(j,tag, env.render_state(state=o.tolist(), label="cl_d%d%s"%(j,tag)))
o=b.copy(); o[103:109]=0.45
print('all', env.render_state(state=o.tolist(), label="cl_all"))
env.close()
