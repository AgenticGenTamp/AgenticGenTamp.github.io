from env_client import make_env
import numpy as np, collections
env = make_env()
L=[];Rr=[];cnt=collections.Counter();bw=[];ow=[]
for seed in range(300):
    obs, info = env.reset(seed=seed)
    cnt[info['object_count']]+=1
    for o in obs.data:
        if o.name=='robot': continue
        x,w=obs.get(o,'x'),obs.get(o,'width')
        L.append(x-w/2);Rr.append(x+w/2)
        if o.name=='target_block': bw.append(w)
        if o.name.startswith('obs'): ow.append(w)
print(min(L),max(Rr),cnt, min(bw),max(bw), np.mean(np.array(bw)<0.25), min(ow),max(ow),np.mean(np.array(ow)<0.25))
env.close()
