from env_client import make_env
import numpy as np
import sys

def p(s,n,fs):
 o=s.get_object_from_name(n)
 return np.array([float(s.get(o,f)) for f in fs])
for direction in sys.argv[1:] or ['xp','xm','yp']:
 e=make_env();s,i=e.reset(seed=1)
 cube=p(s,'cube_0',['x','y','z']); start=cube[:2].copy()
 d={'xp':np.array([1.,0]),'xm':np.array([-1.,0]),'yp':np.array([0,1.]),'ym':np.array([0,-1.])}[direction]
 behind=start-0.52*d
 # approach from high y then side; obstacle island restricts negative x.
 waypoints=[np.array([behind[0],1.5]),behind,start+2*d]
 print('DIRECTION',direction,'START',cube,flush=True)
 for wi,target in enumerate(waypoints):
  for k in range(100):
   xy=p(s,'robot',['pos_base_x','pos_base_y']);a=np.zeros(11,dtype=np.float32)
   delta=target-xy
   a[:2]=np.clip(delta,-.08,.08)
   s,r,t,tr,i=e.step(a)
   if k%5==0 or r!=-1. or t: print(wi,k,'ROBOT',p(s,'robot',['pos_base_x','pos_base_y']), 'CUBE',p(s,'cube_0',['x','y','z']),'R',r,t,tr,flush=True)
   if t or tr or np.linalg.norm(delta)<.015:break
  if t or tr:break
 e.close()
