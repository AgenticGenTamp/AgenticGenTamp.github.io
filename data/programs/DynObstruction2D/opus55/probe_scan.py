import numpy as np
from env_client import make_env
from util import *
env=make_env()
W=3.236
for s in range(301):
    obs,_=env.reset(seed=s)
    d={}
    for o in obs.data:
        d[o.name]={f:obs.get(o,f) for f in ['x','y','theta','width','height']} if o.name!='robot' else None
    b=d['target_block']; t=d['target_surface']
    if t['x']>b['x']: side='L'; gap=b['x']-b['width']/2
    else: side='R'; gap=W-(b['x']+b['width']/2)
    if gap<0.49:
        obst=[k for k in d if k.startswith('obs')]
        print(s,side,round(gap,3),'bw',round(b['width'],3),'bh',round(b['height'],3),'by',round(b['y'],3),'bth',round(b['theta'],2),'rob',rob(obs).round(2)[:2],'nobs',len(obst))
