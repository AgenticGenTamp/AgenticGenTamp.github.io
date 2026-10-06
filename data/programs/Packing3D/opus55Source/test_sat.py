import numpy as np, time
from planner import fast_convex_components_overlap
from kinder.envs.kinematic3d.packing3d import ObjectCentricPacking3DEnv as E
from scipy.spatial.transform import Rotation
f=E._convex_components_overlap
rng=np.random.default_rng(0)
def box():
    he=rng.uniform(0.01,0.05,3); c=rng.uniform(-0.05,0.05,3); R=Rotation.random(random_state=rng.integers(1e9)).as_matrix()
    v=np.array([[sx*he[0],sy*he[1],sz*he[2]] for sx in(-1,1) for sy in(-1,1) for sz in(-1,1)])@R.T+c
    return (v,R.T.copy(),R.T.copy())
bad=0
for i in range(3000):
    a,b=box(),box()
    if f(a,b)!=fast_convex_components_overlap(a,b): bad+=1
print("mismatch",bad)
a,b=box(),box()
t=time.time(); [f(a,b) for _ in range(1000)]; t1=time.time(); [fast_convex_components_overlap(a,b) for _ in range(1000)]; print(t1-t, time.time()-t1)
