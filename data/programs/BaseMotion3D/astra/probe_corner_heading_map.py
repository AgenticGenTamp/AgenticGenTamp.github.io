import numpy as np
from env_client import make_env
for heading in (0.,np.pi/4,-np.pi/4,np.pi/2,-np.pi/2,np.pi-1e-5):
    for xpos in (-.82,-.78,-.74,-.70):
        env=make_env();s,_=env.reset(seed=779)
        for k in range(12):
            a=np.zeros(11,dtype=np.float32);a[2]=np.clip(heading-s[2],-.4,.4);a[0]=np.clip(xpos-s[0],-.4,.4)
            s,*_=env.step(a)
            if abs(s[2]-heading)<1e-5 and abs(s[0]-xpos)<1e-5:break
        for step in [.4]*6+[.2,.1,.05,.025,.0125,.00625,.003125,.0015625,.00078125,.000390625]:
            a=np.zeros(11,dtype=np.float32);a[1]=-step
            s,r,t,tr,i=env.step(a)
        print(round(heading,5),xpos,s[:3].tolist(),flush=True)
        env.close()
