import numpy as np
from env_client import make_env
for heading in np.linspace(-np.pi,np.pi,33)[:-1]:
    env=make_env();s,_=env.reset(seed=779)
    for k in range(20):
        a=np.zeros(11,dtype=np.float32);a[2]=np.clip(heading-s[2],-.4,.4);a[0]=np.clip(s[19]-s[0],-.4,.4)
        n,r,t,tr,i=env.step(a);s=n
        if abs(s[2]-heading)<1e-5 and abs(s[0]-s[19])<1e-5:break
    for step in [.4]*6+[.2,.1,.05,.025,.0125,.00625,.003125,.0015625,.00078125,.000390625]:
        a=np.zeros(11,dtype=np.float32);a[1]=-step
        n,r,t,tr,i=env.step(a);s=n
        if t:break
    print(round(float(heading),5),s[:3].tolist(),bool(t),flush=True)
    env.close()
