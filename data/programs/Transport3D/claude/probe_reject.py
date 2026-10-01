import numpy as np
from probe_util import *
env = make_env()
obs,_ = env.reset(seed=0)
s0 = rs(obs)
print("init robot", fmt(s0))
print("objs", {k:tuple(round(x,3) for x in v) for k,v in objs(obs).items()})
# find a blocked joint motion: drive j2 positive until state stops changing
prev = s0
blocked = None
for i in range(30):
    obs,r,t,tr,info = env.step(act(j2=0.2))
    s = rs(obs)
    if np.allclose(s, prev, atol=1e-9):
        blocked = ("j2",+0.2, i); break
    prev = s
print("j2+ blocked at step", blocked, "state", fmt(prev))
print("info keys", list(info.keys()) if isinstance(info,dict) else info, "reward", r)
# now combine valid base move with blocked joint move
before = rs(obs)
obs,r,t,tr,info = env.step(act(bx=-0.2, j2=0.2))
after = rs(obs)
print("combo bx-0.2 + blocked j2:")
print("  before", fmt(before))
print("  after ", fmt(after))
print("  delta ", fmt(after-before))
# combine blocked joint with gripper close
before = rs(obs)
obs,r,t,tr,info = env.step(act(j2=0.2, g=-1.0))
print("combo blockedj2+close delta", fmt(rs(obs)-before))
# gripper alone
before = rs(obs)
obs,r,t,tr,info = env.step(act(g=-1.0))
print("close alone delta", fmt(rs(obs)-before), "fs", rs(obs)[10], "ga", rs(obs)[11])
before = rs(obs)
obs,r,t,tr,info = env.step(act(g=1.0))
print("open alone delta", fmt(rs(obs)-before), "fs", rs(obs)[10], "ga", rs(obs)[11])
for v in [-1.0,-0.6,-0.4,0.0,0.4,0.6,1.0]:
    obs,r,t,tr,info = env.step(act(g=v))
    print("  g=",v,"fs",round(rs(obs)[10],4),"ga",rs(obs)[11])
# valid base move + valid joint move together (sanity)
before = rs(obs)
obs,r,t,tr,info = env.step(act(bx=-0.2, j4=0.2, g=-1.0))
print("valid combo delta", fmt(rs(obs)-before))
env.close()
