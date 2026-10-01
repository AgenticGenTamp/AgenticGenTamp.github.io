import math
import numpy as np
from env_client import make_env

def run(seed):
 e=make_env(); s,_=e.reset(seed=seed); typ=e.observation_space.get_type
 r=s.get_objects(typ('kin_robot'))[0]; b=s.get_objects(typ('target_block'))[0]; q=s.get_objects(typ('target_surface'))[0]
 bx=float(s.get(b,'x')); by=float(s.get(b,'y')); sx=float(s.get(q,'x'))
 d=1 if sx>bx else -1
 bw=float(s.get(b,'width'))
 ang=0 if d>0 else -math.pi/2
 # Retract and stage overhead first so the arm never sweeps through the cargo.
 tx=bx-.45 if d>0 else bx+bw/2-.12
 ty=by if d>0 else by+.35
 for t in range(25):
  s,*_=e.step(np.array([0,.035,0,-.099,0],float))
 for t in range(70):
  th=float(s.get(r,'theta')); err=(ang-th+math.pi)%(2*math.pi)-math.pi
  a=np.array([np.clip(tx-float(s.get(r,'x')),-.049,.049),0,np.clip(err,-.19,.19),-.099,0],float)
  s,*_=e.step(a)
 for t in range(45):
  s,*_=e.step(np.array([0,np.clip(ty-float(s.get(r,'y')),-.049,.049),0,-.099,0],float))
 if d>0:
  for t in range(11):
   s,*_=e.step(np.array([0,0,0,0,-.019],float))
 for t in range(250):
  s,re,term,tr,_=e.step(np.array([d*.015,0,0,0,0],float))
  if term:
   print(seed,'SUCCESS',t+80,'block',round(float(s.get(b,'x')),2),'surf',round(sx,2),flush=True); e.close(); return True
 print(seed,'fail block',round(float(s.get(b,'x')),2),round(float(s.get(b,'y')),2),'surf',round(sx,2),flush=True); e.close(); return False

for seed in range(12): run(seed)
