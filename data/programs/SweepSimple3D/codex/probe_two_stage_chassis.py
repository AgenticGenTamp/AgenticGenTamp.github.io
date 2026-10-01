"""Rotate/tip the wiper with a diagonal transit, then push due south."""
import math
import numpy as np
from env_client import make_env

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def xy(s,n):
    return np.array([v(s,n,"pos_base_x"),v(s,n,"pos_base_y")]) if n=="robot" else np.array([v(s,n,"x"),v(s,n,"y")])

for angle in (85,86,87):
 for lat in (-.25,):
  e=make_env();s,_=e.reset(seed=0,options={"object_count":1});w0=xy(s,"wiper_0");c0=xy(s,"cube_0")
  d=(c0-w0);d/=np.linalg.norm(d); th=math.radians(angle)
  direction=np.array([d[0]*math.cos(th)-d[1]*math.sin(th),d[0]*math.sin(th)+d[1]*math.cos(th)])
  side=np.array([-direction[1],direction[0]]); goal=w0-.48*direction+lat*side
  for _ in range(24):
   a=np.zeros(11,np.float32);a[:2]=np.clip(.8*(goal-xy(s,"robot")),-.1,.1);s,*_=e.step(a)
  midw=xy(s,"wiper_0").copy();midc=xy(s,"cube_0").copy()
  # Take a collision-free square route around the west side to get north.
  for goal2,limit in ((midw+[-.7,0],30),(midw+[-.7,.55],30),(midw+[0,.55],30)):
   for _ in range(limit):
    a=np.zeros(11,np.float32);a[:2]=np.clip(.8*(goal2-xy(s,"robot")),-.1,.1);s,r,t,tr,_=e.step(a)
  # Follow the wiper south rather than aiming at a stale absolute endpoint.
  for _ in range(100):
   w=xy(s,"wiper_0"); goal2=w+[0,.18]
   a=np.zeros(11,np.float32);a[:2]=np.clip(.8*(goal2-xy(s,"robot"))+[0,-.035],-.06,.06);s,r,t,tr,_=e.step(a)
  print(angle,lat,"midw",np.round(midw-w0,3),"midc",np.round(midc-c0,3),"finalw",np.round(xy(s,"wiper_0")-w0,3),"finalc",np.round(xy(s,"cube_0")-c0,3),"r",r,flush=True)
  e.close()
