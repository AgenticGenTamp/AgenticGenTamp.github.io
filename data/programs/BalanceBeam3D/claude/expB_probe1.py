import numpy as np
from env_client import make_env
np.set_printoptions(precision=4,suppress=True)
env=make_env(); obs,_=env.reset(seed=0)
s=np.asarray(env.get_state(),dtype=float)
print("state len",s.shape, "obs len", obs.shape)
for name,bi in [("large",0),("s1",54),("s2",70),("ss",38)]:
    p=obs[bi:bi+3]
    idx=[i for i in range(len(s)) if abs(s[i]-p[0])<1e-9]
    print(name,p,"matches at",idx[:5])
print("s head",s[:40])
env.close()
