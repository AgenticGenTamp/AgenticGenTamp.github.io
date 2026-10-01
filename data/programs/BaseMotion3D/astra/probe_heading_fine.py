import numpy as np
from env_client import make_env
for heading in np.linspace(-.1,.2,31):
    env=make_env();s,_=env.reset(seed=779)
    a=np.zeros(11,dtype=np.float32);a[2]=heading;a[0]=-.4;s,*_=env.step(a)
    a[:]=0;a[0]=s[19]-s[0];s,*_=env.step(a)
    for step in [.4]*5+[.2,.1,.05,.025,.0125,.00625,.003125,.0015625,.00078125,.000390625,.0001953125]:
        a[:]=0;a[1]=-step
        s,r,t,tr,i=env.step(a)
        if t:break
    print(round(float(heading),5),s[:3].tolist(),bool(t),flush=True)
    env.close()
