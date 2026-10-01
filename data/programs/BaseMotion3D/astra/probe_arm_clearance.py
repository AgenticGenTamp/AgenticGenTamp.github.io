import numpy as np
from env_client import make_env
cases=['original','zero']+[(j,v) for j in range(3,10) for v in (-1.2,-.4,.4,1.2)]
for case in cases:
    env=make_env();s,_=env.reset(seed=560);initial=s.copy()
    desired=s[3:10].copy()
    if case=='zero':desired[:]=0
    elif case!='original':desired[case[0]-3]+=case[1]
    for k in range(15):
        a=np.zeros(11,dtype=np.float32);a[3:10]=np.clip(desired-s[3:10],-.4,.4);a[0]=np.clip(s[19]-s[0],-.4,.4)
        n,r,t,tr,i=env.step(a)
        if np.max(np.abs(n-s))<1e-6:s=n;break
        s=n
    reached=s[3:10].copy()
    for step in [.4]*5+[.2,.1,.05,.025,.0125,.00625,.003125,.0015625,.00078125,.000390625,.0001953125]:
        a=np.zeros(11,dtype=np.float32);a[1]=-step
        s,r,t,tr,i=env.step(a)
        if t:break
    print(str(case),s[:3].tolist(),bool(t),'joints',reached.tolist(),flush=True)
    env.close()
