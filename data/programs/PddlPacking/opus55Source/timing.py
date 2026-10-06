import sys, time
import numpy as np
import approach as A
orig = A.GeneratedApproach.replan
slow = []
def rp(self, flat):
    t=time.time(); r=orig(self, flat)
    dt=time.time()-t
    print(f"  replan t={self.elapsed():.1f} dt={dt:.2f} len={None if r is None else len(r[0])}", flush=True)
    if dt > 5: slow.append(np.array(flat))
    return r
A.GeneratedApproach.replan = rp
import test_approach as T
k=int(sys.argv[1])
for s in [int(x) for x in sys.argv[2].split(',')]:
    print(s, T.run(s,k), flush=True)
    if slow: np.save(f'slow_{k}_{s}.npy', np.array(slow)); slow.clear()
