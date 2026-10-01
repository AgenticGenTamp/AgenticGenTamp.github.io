import numpy as np
from probe_act import rollout, FEATS
np.set_printoptions(precision=4,suppress=True)
print("dim | "+" ".join(f"{f.replace('pos_','')[:7]:>8s}" for f in FEATS))
for dim in range(10):
    a=[0.0]*10+[0.0]; a[dim]=0.1
    h,r,term,trunc,info=rollout(a,steps=50)
    d1=h[1]-h[0]; d50=h[-1]-h[0]
    print(f"d{dim} s1  "+" ".join(f"{v:8.4f}" for v in d1))
    print(f"d{dim} s50 "+" ".join(f"{v:8.4f}" for v in d50)+f"  | rewsum {r.sum():.1f} term {term} trunc {trunc}")
