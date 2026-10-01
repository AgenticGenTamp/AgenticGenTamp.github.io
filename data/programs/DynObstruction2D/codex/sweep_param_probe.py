import math
import sys
import numpy as np
from env_client import make_env


def angerr(want, have):
    return (want-have+math.pi) % (2*math.pi)-math.pi


seed = int(sys.argv[1])
angle = float(sys.argv[2])
xoff = float(sys.argv[3])
yoff = float(sys.argv[4])
gap = float(sys.argv[5])
speed = float(sys.argv[6])
env = make_env(); s, info = env.reset(seed=seed); typ = env.observation_space.get_type
r = s.get_objects(typ("kin_robot"))[0]; b = s.get_objects(typ("target_block"))[0]
q = s.get_objects(typ("target_surface"))[0]
g = lambda o, f: float(s.get(o, f))
direction = 1 if g(q, "x") >= g(b, "x") else -1
terminated = False

def step(action):
    global s, terminated
    s, rew, term, trunc, info = env.step(np.asarray(action, dtype=env.action_space.dtype))
    terminated = terminated or term
    return term or trunc

# Raise, align at minimum arm length, then lower to an offset from block center.
for _ in range(30):
    a = [0, np.clip(1.1-g(r,"y"),-.049,.049), 0, -.099,
         np.clip(gap-g(r,"finger_gap"),-.019,.019)]
    if step(a): break
bx0, by0 = g(b,"x"), g(b,"y")
tx = bx0 + direction*xoff
for _ in range(80):
    a = [np.clip(tx-g(r,"x"),-.049,.049), 0,
         np.clip(angerr(angle,g(r,"theta")),-.19,.19),-.099,
         np.clip(gap-g(r,"finger_gap"),-.019,.019)]
    if step(a): break
    if abs(tx-g(r,"x"))<.01 and abs(angerr(angle,g(r,"theta")))<.02: break
ty = by0+yoff
for _ in range(40):
    a = [0,np.clip(ty-g(r,"y"),-.049,.049),0,-.099,
         np.clip(gap-g(r,"finger_gap"),-.019,.019)]
    if step(a): break
    if abs(ty-g(r,"y"))<.01: break
print("contact",seed,angle,xoff,yoff,gap,"r",round(g(r,"x"),2),round(g(r,"y"),2),
      "b",round(g(b,"x"),2),round(g(b,"y"),2),round(g(b,"theta"),2),flush=True)
for i in range(300):
    if step([direction*speed,0,0,0,0]): break
    if i % 50 == 49:
        print(" push",i+1,"r",round(g(r,"x"),2),"b",round(g(b,"x"),2),
              round(g(b,"y"),2),round(g(b,"theta"),2),flush=True)
print("RESULT",bool(terminated),i+1,"b",round(g(b,"x"),3),round(g(b,"y"),3),
      "surf",round(g(q,"x"),3),flush=True)
env.close()
