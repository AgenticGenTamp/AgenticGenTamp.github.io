import sys, math, numpy as np
import pybullet as p
from robocode.environments.rovers_variable_count_env import RoversVariableCountEnv
from robocode.environments.ss_pybullet import *
from approach import GeneratedApproach
seed, T, r = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
env = RoversVariableCountEnv(design_counts=[1,2,3,4], eval_counts=[1,2,3,4])
state, info = env.reset(seed=seed)
a = GeneratedApproach(env.action_space, env.observation_space, {})
a.reset(state, info)
for k in range(T):
    state, *_ = env.step(a.get_action(state))
b = env.current_backend
names = {}
with b.client():
    for i in range(p.getNumBodies()):
        bid = p.getBodyUniqueId(i); names[bid] = p.getBodyInfo(bid)[1].decode()
    for i,o in enumerate(b._objectives): names[o] = f"objective{i}"
    names[b._lander]="lander"
    for i,ro in enumerate(b._rovers): names[ro]=f"rover{i}"
    for i,ob in enumerate(b._obstacle_bodies): names[ob]=f"obstacle{i}"
    for i,ob in enumerate(b._mounds): names[ob]=f"mound{i}"
    x,y,th = get_joint_positions(b._rovers[r], b._base_joints[r])
    print("pose", round(x,3), round(y,3), round(th,3), "other", [round(v,2) for v in get_joint_positions(b._rovers[1-r], b._base_joints[1-r])])
    cx = x - 0.087*math.cos(th) + 0.0125*math.sin(th); cy = y - 0.087*math.sin(th) - 0.0125*math.cos(th)
    tg = [(f"obj{i}", o) for i,o in enumerate(b._objectives)] + [("lander", b._lander)]
    for nm, o in tg:
        pt = get_point(o)
        hit = p.rayTest([cx,cy,0.2992], list(pt))[0]
        vis = b.image_visible(r, o) if nm!="lander" else b.com_visible(r)
        d3 = math.dist((cx,cy,0.2992), pt)
        print(nm, np.round(pt,2), "vis", int(vis), "d3", round(d3,3), "hit", names.get(hit[0], hit[0]), round(hit[2],3))
