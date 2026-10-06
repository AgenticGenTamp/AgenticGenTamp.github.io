import time, numpy as np
import planner
from planner import TwinPlanner, jac_ik, yaw_quat, quat_mul
from pybullet_helpers.geometry import Pose
from blas1 import limit_blas_threads; limit_blas_threads(1)
tp = TwinPlanner(2); arm = tp.arm
rng = np.random.default_rng(0); home = list(arm.get_joint_positions())
its=[]; orig=planner._p.calculateJacobian; cnt=[0]
def wrap(*a, **k):
    cnt[0]+=1; return orig(*a, **k)
planner._p.calculateJacobian = wrap
T=0
for i in range(300):
    pos = (rng.uniform(0.15,0.6), rng.uniform(-0.35,0.35), rng.uniform(0.12,0.4))
    pose = Pose(pos, quat_mul(yaw_quat(rng.uniform(-3.14,3.14)), tp.down))
    seed = home if i%2 else list(np.array(home)+np.r_[rng.normal(0,0.3,7),np.zeros(len(home)-7)])
    arm.set_joints(seed); cnt[0]=0; t=time.time()
    try: jac_ik(arm, pose); ok=True
    except Exception as e: ok=False
    T+=time.time()-t; its.append((cnt[0], ok))
a=np.array([c for c,o in its if o]); print("ok iters mean",a.mean(),"max",a.max(),"pct90",np.percentile(a,90)); print("fail iters",[c for c,o in its if not o]); print("ms per ik",T/300*1000)
