"""Determine whether the discovered contact can retain the rod."""
import sys
import numpy as np
from env_client import make_env

grip = float(sys.argv[1]) if len(sys.argv) > 1 else -1.
env = make_env(); state, _ = env.reset(seed=1)
robot = state.get_object_from_name("robot")
rod = state.get_object_from_name("cuboid_0")
qgoal = np.array([-3.1003, -2.4, .2638, -.4262, -1.5708, -.1238, 2.8392])
def v(o,f): return float(state.get(o,f))
def pos(o): return np.array([v(o,f) for f in ("x","y","z")])
origin=pos(rod)

def step(goal,g):
 global state
 a=np.zeros(11)
 b=np.array([v(robot,f) for f in ("pos_base_x","pos_base_y","pos_base_rot")])
 e=goal-b; e[2]=(e[2]+np.pi)%(2*np.pi)-np.pi; a[:3]=np.clip(e/.87,-.1,.1)
 q=np.array([v(robot,f"pos_arm_joint{i}") for i in range(1,8)])
 qe=(qgoal-q+np.pi)%(2*np.pi)-np.pi; a[3:10]=np.clip(.5*qe,-.1,.1); a[-1]=g
 state,*_=env.step(a.astype(np.float32))

rear=np.array([origin[0]-.80,origin[1]+.10,0.])
touch=np.array([origin[0]-.70,origin[1]+.10,0.])
for _ in range(150): step(rear,1.)
for _ in range(25): step(touch,grip)
p_touch=pos(rod).copy()
for _ in range(25): step(touch,grip)
for _ in range(45): step(rear,grip)
p_final=pos(rod)
q=np.array([v(robot,f"pos_arm_joint{i}") for i in range(1,8)])
b=np.array([v(robot,f) for f in ("pos_base_x","pos_base_y","pos_base_rot")])
print("grip",grip,"origin",np.round(origin,6),"touch",np.round(p_touch,6),
      "final",np.round(p_final,6),"delta",np.round(p_final-origin,6),
      "base",np.round(b,6),"q",np.round(q,6),flush=True)
env.close()
