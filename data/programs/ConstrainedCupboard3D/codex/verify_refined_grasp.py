"""Reproduce refined contact, then move away to test attachment."""
import numpy as np
import sys
from env_client import make_env
env=make_env(); state,_=env.reset(seed=1)
r=state.get_object_from_name("robot"); rod=state.get_object_from_name("cuboid_0")
qg=np.array([-3.0175,-2.3524,-1.1879,-.314,-1.5966,.6707,-2.7101])
def v(o,f): return float(state.get(o,f))
def p(o): return np.array([v(o,f) for f in ("x","y","z")])
p0=p(rod)
open_approach = len(sys.argv) > 1
close_dy = float(sys.argv[2]) if len(sys.argv) > 2 else .1
def go(g,grip):
 global state
 a=np.zeros(11); b=np.array([v(r,f) for f in ("pos_base_x","pos_base_y","pos_base_rot")])
 e=g-b; e[2]=(e[2]+np.pi)%(2*np.pi)-np.pi; a[:3]=np.clip(e/.87,-.1,.1)
 q=np.array([v(r,f"pos_arm_joint{i}") for i in range(1,8)])
 e=(qg-q+np.pi)%(2*np.pi)-np.pi; a[3:10]=np.clip(.5*e,-.1,.1); a[-1]=grip
 state,*_=env.step(a.astype(np.float32))
start=np.array([p0[0]-.8,p0[1]-.5,0.])
for _ in range(150): go(start,1.)
for dy in np.arange(-.5,.101,.1):
 for _ in range(10): go(np.array([p0[0]-.8,p0[1]+dy,0.]),1. if open_approach else 0.)
pt=p(rod).copy()
# Recenter laterally while open.
for _ in range(15): go(np.array([p0[0]-.8,p0[1]+close_dy,0.]),1.)
p_preclose=p(rod).copy()
# Close in place before attempting the pull.
for _ in range(15): go(np.array([p0[0]-.8,p0[1]+close_dy,0.]),0.)
p_closed=p(rod).copy()
# With fingers closed, move another 20 cm along y and 10 cm backward in x.
carry=(np.array([p0[0]-.8,p0[1]-.1,0.]) if open_approach else
       np.array([p0[0]-.9,p0[1]+.3,0.]))
for _ in range(45): go(carry,0.)
pc=p(rod).copy()
# Open and leave the object.
for _ in range(15): go(carry,1.)
print("open_approach",open_approach,"close_dy",close_dy,"origin",np.round(p0,6),"touch",np.round(pt,6),
 "preclose",np.round(p_preclose,6),"closed",np.round(p_closed,6),"carry",np.round(pc,6),
 "touch_delta",np.round(pt-p0,6),"carry_delta",np.round(pc-pt,6),
 "base",np.round([v(r,f) for f in ("pos_base_x","pos_base_y","pos_base_rot")],6),flush=True)
env.close()
