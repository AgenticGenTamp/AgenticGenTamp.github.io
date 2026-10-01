"""Scan base yaw at the closest reachable base point for a fixed posture."""
import sys
import numpy as np
from env_client import make_env

q = np.asarray([float(x) for x in sys.argv[1].split(",")])
target = sys.argv[2] if len(sys.argv) > 2 else "box0"
env = make_env(); s, _ = env.reset(seed=1)

def g(n, f): return float(s.get(s.get_object_from_name(n), f))

def act(goal, yaw, grip):
    global s
    a=np.zeros(11,np.float32)
    a[0]=np.clip(goal[0]-g("robot","pos_base_x"),-.2,.2)
    a[1]=np.clip(goal[1]-g("robot","pos_base_y"),-.2,.2)
    # Rotation itself is unbounded/wrapped, but use relative shortest-ish motion.
    a[2]=np.clip(yaw-g("robot","pos_base_rot"),-.2,.2)
    cur=np.asarray([g("robot","joint_%d"%i) for i in range(1,8)])
    a[3:10]=np.clip(q-cur,-.2,.2); a[10]=grip
    s,*_=env.step(a)

tx,ty=g(target,"pose_x"),g(target,"pose_y")
base=(np.clip(tx,-1,1),np.clip(ty,-1,1))
for yaw in np.linspace(-3.0,3.0,31):
    for _ in range(32): act(base,yaw,1.)
    act(base,yaw,-1.)
    if g("robot","grasp_active")>.5:
        print("HIT q",[g("robot","joint_%d"%i) for i in range(1,8)],
              "base",[g("robot","pos_base_x"),g("robot","pos_base_y"),g("robot","pos_base_rot")],
              "object-minus-base",[tx-g("robot","pos_base_x"),ty-g("robot","pos_base_y")])
        env.close();raise SystemExit
print("MISS",q.tolist());env.close()
