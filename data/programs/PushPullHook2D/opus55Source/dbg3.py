import sys, numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import *
env = KinderGeom2DEnv("kinder/PushPullHook2D-v0")
s=int(sys.argv[1])
obs, info = env.reset(seed=s)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
d = ap._parse(obs)
c = ap._grasp_candidates(d)
print("ncands", len(c))
from collections import Counter
fails = Counter()
for key,b in c:
    T = compose(inverse(b), d["hook"])
    for opt in ap._push_options(d, T, d["arm_max"]):
        fi,t,B,u,D,hth = opt
        pre = (B[0]-u[0]*0.09, B[1]-u[1]*0.09, B[2]); end=(B[0]+u[0]*D, B[1]+u[1]*D, B[2])
        r1 = ap._pose_valid(d, pre, T, d["arm_max"], d["btn"], 0.04)
        r2 = ap._pose_valid(d, B, T, d["arm_max"], d["btn"], 0, check_btn=False)
        r3 = ap._pose_valid(d, end, T, d["arm_max"], d["btn"], 0, check_btn=False)
        fails[(key[0], fi, r1, r2, r3)] += 1
for k,v in sorted(fails.items()): print(k, v)
print("----")
import approach
def reasons(d, B, T, arm, btn, margin):
    x,y,th = B; rad=d["rad"]; out=[]
    if not (rad+0.005 < x < 3.5-rad-0.005): out.append("x")
    if not (rad+0.005 < y < 1.25-rad-0.003): out.append("y%.2f"%y)
    H = compose(B,T)
    wx,wy = ap.hook.hull_world((np.array(H[0]),np.array(H[1]),np.array(H[2])))
    if wx.min()<0.004 or wx.max()>3.496: out.append("hx")
    if wy.min()<0.004 or wy.max()>2.496: out.append("hy%.2f,%.2f"%(wy.min(),wy.max()))
    g = np.array([x+arm*np.cos(th), y+arm*np.sin(th)])
    if np.hypot(*(g-btn)) < d["br"]+0.045: out.append("grip")
    if ap.hook.dist_world(H, np.array([btn[0]]), np.array([btn[1]]))[0] < d["br"]+margin: out.append("hookbtn")
    return out
R = Counter()
for key,b in c:
    T = compose(inverse(b), d["hook"])
    for opt in ap._push_options(d, T, d["arm_max"]):
        fi,t,B,u,D,hth = opt
        pre = (B[0]-u[0]*0.09, B[1]-u[1]*0.09, B[2])
        R[(fi, tuple(reasons(d, pre, T, d["arm_max"], d["btn"], 0.04)))] += 1
for k,v in sorted(R.items(), key=lambda z:-z[1])[:25]: print(k, v)
