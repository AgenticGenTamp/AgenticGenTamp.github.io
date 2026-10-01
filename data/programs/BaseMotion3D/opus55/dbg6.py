import numpy as np
from env_client import make_env
exec(open('dbg5.py').read().split("print('rot sweep')")[0].split('for x in')[0].replace('env=make_env()','env=make_env()'))
tests={'none':{}, 'j2+':{4:0.8}, 'j2-':{4:-0.8},'j4+':{6:0.8},'j4-':{6:-0.8},'j3+':{5:0.8},'j1+':{3:0.8},'j1-':{3:-0.8},'j6+':{8:0.8},'j6-':{8:-0.8},'j1pi':{3:3.14}}
for name,dj in tests.items():
    o,_=env.reset(seed=0)
    o,_=go(o,np.array([0.4,-1.0]))
    for j,v in dj.items():
        n=int(np.ceil(abs(v)/0.4))
        for _ in range(n):
            a=np.zeros(11); a[j]=v/n; o,*_=env.step(a)
    j=o[3:10].copy()
    o,_=go(o,np.array([0.4,-3.0]))
    print(name,np.round(j,2),np.round(o[:3],3))
