import sys
import numpy as np
from env_client import make_env
import approach as A
sd=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=sd)
s=np.asarray(obs,dtype=float)
ap=A.GeneratedApproach(env.action_space, env.observation_space, {})
cands,v0=ap._candidates(s,held=False)
print("n cands",len(cands))
hook=ap._hook_of(s)
mask=ap._base_mask(s,hook)
comp=ap._component(mask,ap._cell(s[:2]))
reach=[c for c in cands if comp[ap._cell(c['S'])]]
print("reachable",len(reach))
from collections import Counter
print(Counter([(c['mode'],round(c['sig'])) for c in cands]))
print(Counter([(c['mode'],round(c['sig'])) for c in reach]))
for c in reach[:3]:
    print(" ",c['mode'],c['sig'],'d',round(c['d'],2),'gamma',round(np.degrees(c['gamma']),1),'S',np.round(c['S'],3),'rp',np.round(c['rp'],3),'cost',round(c['cost'],1))
env.close()
