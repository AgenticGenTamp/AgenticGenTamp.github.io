import numpy as np
from env_client import make_env
selected=[]
for seed in range(1000000):
    p=np.random.default_rng(seed).uniform(-2,2,2)
    if -.65<p[0]<-.5 and p[1]<-1.959:
        selected.append((seed,p))
        if len(selected)==15:break
print('SELECTED',[(k,p.tolist()) for k,p in selected],flush=True)
for seed,p in selected:
    answers=[]
    for heading in (np.pi/4,0.,.2,.4,.6,.8,1.,1.2):
        env=make_env();s,_=env.reset(seed=seed)
        assert np.allclose(s[19:21],p), (seed,p,s[19:21])
        success=False;which=-1;count=0
        for which,offset in enumerate(([-.0353546,.0353546],[0,.049999],[-.049999,0])):
            aim=s[19:21]+offset
            for k in range(10):
                a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(aim-s[:2],-.4,.4);a[2]=np.clip(heading-s[2],-.4,.4)
                n,r,t,tr,i=env.step(a);count+=1
                changed=np.max(np.abs(n[:3]-s[:3]))>1e-7;s=n
                if t:success=True;break
                if not changed:break
            if success:break
        answers.append([round(heading,4),success,which,count,s[:3].tolist()]);env.close()
    print('RESULT',seed,p.tolist(),answers,flush=True)
