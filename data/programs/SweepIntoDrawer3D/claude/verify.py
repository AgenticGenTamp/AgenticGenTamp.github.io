import numpy as np
from env_client import make_env
env=make_env(); obs,_=env.reset(seed=0); obs=np.asarray(obs,float)
b=obs.copy()
for c in range(5): b[c*16+0]=20; b[c*16+1]=20; b[c*16+2]=20
b[147]=20;b[148]=20;b[149]=20; b[125]=4.0; b[126]=4.0
ys=[0.665,0.0,-0.665]
o=b.copy(); o[103:106]=0.45
for i,y in enumerate(ys): o[i*16+0]=1.11; o[i*16+1]=y; o[i*16+2]=0.12
print(env.render_state(state=o.tolist(), label="ver_low"))
o=b.copy(); o[106:109]=0.45
for i,y in enumerate(ys): o[i*16+0]=1.11; o[i*16+1]=y; o[i*16+2]=0.33
print(env.render_state(state=o.tolist(), label="ver_up"))
env.close()
