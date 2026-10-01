import numpy as np
from probe_act import rollout, FEATS
np.set_printoptions(precision=4,suppress=True)
# linearity / limits: run dim3 (joint1) and dim0 (base x) long
for dim,val,steps in [(3,0.1,200),(0,0.1,120),(3,-0.1,200),(4,0.1,200)]:
    a=[0.0]*10+[0.0]; a[dim]=val
    h,r,term,trunc,info=rollout(a,steps=steps)
    fi = 0 if dim<3 else dim-3+3
    col=h[:,fi]
    ds=np.diff(col)
    print(f"dim{dim} val{val}: feat={FEATS[fi]} start={col[0]:.4f} end={col[-1]:.4f}")
    print("  per-step deltas idx0-9:",np.round(ds[:10],4))
    print("  deltas @20,40,60,100,150,199:",np.round([ds[min(i,len(ds)-1)] for i in [20,40,60,100,150,199]],4))
    print("  vals @ 25,50,100,150,end:",np.round([col[min(i,len(col)-1)] for i in [25,50,100,150,len(col)-1]],4))
