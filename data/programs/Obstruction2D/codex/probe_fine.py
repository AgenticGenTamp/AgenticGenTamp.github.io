import numpy as np
from env_client import make_env

for ext in [0,.1]:
 e=make_env(); s,_=e.reset(seed=0); T=e.observation_space
 r=s.get_objects(T.get_type("crv_robot"))[0]; b=s.get_objects(T.get_type("target_block"))[0]
 tx=float(s.get(b,"x"))+.05
 for _ in range(10):
  dx=np.clip(tx-float(s.get(r,"x")),-.05,.05); s,*_=e.step(np.array([dx,0,0,0,1],np.float32))
 if ext: s,*_=e.step(np.array([0,0,0,ext,1],np.float32))
 threshold=float(s.get(b,"y"))+float(s.get(b,"height"))+float(s.get(r,"arm_joint"))+float(s.get(r,"gripper_width"))
 while float(s.get(r,"y")) > threshold+.011: s,*_=e.step(np.array([0,-.01,0,0,1],np.float32))
 ys=[]
 for _ in range(30):
  y0=float(s.get(r,"y")); s,*_=e.step(np.array([0,-.001,0,0,1],np.float32)); y1=float(s.get(r,"y")); ys.append((y0,y1))
  if y0==y1: break
 print("ext",ext,"pred",threshold,"last",ys[-3:])
 e.close()
