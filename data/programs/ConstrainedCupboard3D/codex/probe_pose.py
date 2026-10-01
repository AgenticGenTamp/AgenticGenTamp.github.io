"""Move to one rod and test a kinematically plausible floor pose."""
import numpy as np
from env_client import make_env

e = make_env(); s, _ = e.reset(seed=1)
rob = s.get_object_from_name("robot"); rod = s.get_object_from_name("cuboid_1")
def g(o, f): return float(s.get(o, f))
home = np.array([g(rob, f"pos_arm_joint{i}") for i in range(1, 8)])
floor = np.array([-.075, -2.568, -2.346, -3.0, 2.389, 2.539, -.737])
orig = np.array([g(rod, f) for f in ("x", "y", "z")])
for k in range(180):
    a = np.zeros(11, np.float32); a[10] = 1
    bx, by = g(rob,"pos_base_x"), g(rob,"pos_base_y")
    if k < 10:
        a[0] = np.clip((orig[0]-.4-bx)/.87,-.1,.1)
        a[1] = np.clip((orig[1]-by)/.87,-.1,.1)
    elif k < 130:
        q=np.array([g(rob,f"pos_arm_joint{i}") for i in range(1,8)])
        err=(floor-q+np.pi)%(2*np.pi)-np.pi
        a[3:10]=np.clip(.4*err,-.1,.1)
    else:
        a[10]=0
    s,r,t,tr,info=e.step(a)
    if k%10==0:
        p=np.array([g(rod,f) for f in ("x","y","z")])
        print(k,"base",round(g(rob,"pos_base_x"),2),round(g(rob,"pos_base_y"),2),"move",np.round(p-orig,3),"r",r,flush=True)
e.close()
