import sys, numpy as np, approach
from env_client import make_env
env=make_env(); ap=approach.GeneratedApproach(env.action_space, env.observation_space, {})
kw={'options':{'object_count':int(sys.argv[2])}} if len(sys.argv)>2 else {}
obs,info=env.reset(seed=int(sys.argv[1]),**kw); ap.reset(obs,info)
out=[]
for t in range(1000):
    a=ap.get_action(obs)
    obs,r,term,trunc,info=env.step(a)
    bl=np.max(np.abs(a[:2])); br=abs(a[2]); j=np.max(np.abs(a[3:10])); g=a[10]
    c=('T' if bl>1e-3 else '')+('R' if br>1e-3 else '')+('J' if j>0.05 else ('j' if j>1e-3 else ''))+('O' if g>0 else ('C' if g<0 else ''))
    out.append(c or '_')
    if term: break
print(t+1, ' '.join(out))
