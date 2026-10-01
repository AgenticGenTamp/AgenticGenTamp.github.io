"""Dense arm-only sweep around the known low wrist posture at safe radius."""
import math
import numpy as np
from env_client import make_env

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def p(s,n): return np.array([v(s,n,f) for f in "xyz"])
def b(s): return np.array([v(s,"robot","pos_base_x"),v(s,"robot","pos_base_y")])
def q(s): return np.array([v(s,"robot",f"pos_arm_joint{i}") for i in range(1,8)])
def ae(a,b): return (a-b+math.pi)%(2*math.pi)-math.pi

e=make_env(); s,_=e.reset(seed=0,options={"object_count":1}); w=p(s,"wiper_0")
# Approach radially from the north; the entire chassis path remains outside 36cm.
goal=w[:2]+np.array([0.,.365]); yaw=-math.pi/2; q0=q(s)
for _ in range(55):
 a=np.zeros(11,np.float32); a[:2]=np.clip(.7*(goal-b(s)),-.08,.08)
 a[2]=np.clip(.7*ae(yaw,v(s,"robot","pos_base_rot")),-.1,.1)
 a[3:10]=np.clip(.7*(q0-q(s)),-.1,.1); a[10]=1; s,*_=e.step(a)
for _ in range(5):
 a=np.zeros(11,np.float32);a[10]=1;s,*_=e.step(a)
print("POSE base",b(s).round(5).tolist(),"yaw",round(v(s,"robot","pos_base_rot"),5),
      "w",p(s,"wiper_0").round(5).tolist(),"dist",round(np.linalg.norm(b(s)-p(s,"wiper_0")[:2]),5),flush=True)

# q1 scans laterally while q2/q4 span the low-reaching envelope. Other wrist
# joints stay at the best visually identified orientation.
for q2 in (.5,.9,1.3):
 for q4 in (-2.9,-2.55,-2.2,-1.9):
  for q1 in (-1.1,-.55,0.,.55,1.1):
   target=np.array([q1,q2,2.37,q4,.02,-.8,1.57]); before=p(s,"wiper_0"); qb=q(s).copy();prev=before.copy();mx=0
   for k in range(12):
    a=np.zeros(11,np.float32);a[3:10]=np.clip(.8*(target-q(s)),-.1,.1);a[10]=1
    s,*_=e.step(a);now=p(s,"wiper_0");mx=max(mx,float(np.linalg.norm(now-prev)));prev=now
   d=p(s,"wiper_0")-before; dist=np.linalg.norm(b(s)-p(s,"wiper_0")[:2])
   if np.linalg.norm(d)>.0015 or mx>.0015:
    print("CONTACT base",b(s).round(5).tolist(),"yaw",round(v(s,"robot","pos_base_rot"),5),
          "q_before",qb.round(5).tolist(),"target",target.tolist(),"actual",q(s).round(5).tolist(),
          "w_before",before.round(5).tolist(),"delta",d.round(5).tolist(),
          "max_step",round(mx,5),"base_dist",round(dist,5),flush=True)
   if dist <= .35:
    print("INVALID DIST",dist,flush=True); raise SystemExit
e.close()
