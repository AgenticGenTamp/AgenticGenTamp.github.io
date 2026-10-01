import numpy as np
from probe_act import rollout, FEATS
for dim in range(3,10):
    for val in (0.1,-0.1):
        a=[0.0]*10+[0.0]; a[dim]=val
        h,_,_,_,_=rollout(a,steps=400)
        fi=dim; col=h[:,fi]
        # find where it stops moving
        ds=np.abs(np.diff(col))
        stop=np.argmax(ds<1e-4) if (ds<1e-4).any() else -1
        print(f"dim{dim}({FEATS[fi]}) val{val:+.1f}: start={col[0]:+.3f} end={col[-1]:+.3f} stalled_at_step={stop if stop>0 else 'never'}")
