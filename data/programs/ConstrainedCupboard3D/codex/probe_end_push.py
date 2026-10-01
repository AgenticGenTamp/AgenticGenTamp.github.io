"""Measure x push from refined downward pose on seed 1."""
import sys
import numpy as np
from env_client import make_env

yoff = float(sys.argv[1])
dq1 = float(sys.argv[2])
xoff = float(sys.argv[3]) if len(sys.argv) > 3 else -.8
direction = int(sys.argv[4]) if len(sys.argv) > 4 else 1
env = make_env(); state, _ = env.reset(seed=1)
robot = state.get_object_from_name("robot")
rod = state.get_object_from_name("cuboid_0")
qg = np.array([-3.0175+dq1,-2.3524,-1.1879,-.314,-1.5966,.6707,-2.7101])
def v(o,f): return float(state.get(o,f))
def rp(): return np.array([v(rod,f) for f in ("x","y","z")])
p0=rp()
best_reward=-1e9; terminal=None
def go(goal, grip=0.):
 global state,best_reward,terminal
 a=np.zeros(11,np.float32)
 b=np.array([v(robot,f) for f in ("pos_base_x","pos_base_y","pos_base_rot")])
 e=goal-b; e[2]=(e[2]+np.pi)%(2*np.pi)-np.pi
 a[:3]=np.clip(e/.87,-.1,.1)
 q=np.array([v(robot,f"pos_arm_joint{i}") for i in range(1,8)])
 qe=(qg-q+np.pi)%(2*np.pi)-np.pi
 a[3:10]=np.clip(.5*qe,-.1,.1); a[-1]=grip
 state,reward,terminated,truncated,info=env.step(a)
 best_reward=max(best_reward,float(reward))
 if terminated or truncated:
  terminal=(terminated,truncated,info)
# Establish low pose clear behind and to side, then align to chosen end/offset.
start=np.array([p0[0]+xoff,p0[1]-.5,0.])
for _ in range(125): go(start,1.)
contact=np.array([p0[0]+xoff,p0[1]+yoff,0.])
for _ in range(30): go(contact,1.)
for _ in range(10): go(contact,0.)
# Push through rod in +x; feedback in y follows rod to preserve offset.
best=-9.; bestrow=None
for k in range(75):
 ry=v(rod,"y")
 target_x = 1.20 if direction > 0 else -0.60
 go(np.array([target_x,ry+yoff,0.]),0.)
 d=rp()-p0
 if d[0]>best: best=d[0]; bestrow=(k, rp().copy(), np.array([v(robot,f) for f in ("pos_base_x","pos_base_y","pos_base_rot")]))
print("RESULT",yoff,dq1,xoff,direction,"origin",np.round(p0,5),"delta",np.round(rp()-p0,5),"bestx",round(best,5),"best",bestrow,"best_reward",best_reward,"terminal",terminal,"qactual",np.round([v(robot,f"pos_arm_joint{i}") for i in range(1,8)],5),flush=True)
env.close()
