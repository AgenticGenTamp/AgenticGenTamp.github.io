from env_client import make_env
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def run(direction):
 e=make_env();s,i=e.reset(seed=1,options={'object_count':1});ro=s.get_object_from_name('robot');cu=s.get_object_from_name('cube_0')
 def xy(o):return np.array([s.get(o,'x'),s.get(o,'y')])
 orig=xy(cu);d=np.array(direction,float);theta=np.arctan2(d[1],d[0]);goalq=np.array([0,2.4,np.pi,.3,0,-.873,np.pi/2])
 # Approach outside the floor-object cluster before lowering.
 setup=orig-.8*d
 for k in range(500):
  a=np.zeros(11);q=np.array([s.get(ro,'pos_arm_joint%d'%j) for j in range(1,8)])
  b=np.array([s.get(ro,'pos_base_x'),s.get(ro,'pos_base_y')]);th=s.get(ro,'pos_base_rot')
  a[2]=np.clip((theta-th+np.pi)%(2*np.pi)-np.pi,-.1,.1)
  a[10]=1
  if k<70:a[:2]=np.clip(setup-b,-.06,.06)
  else:a[3:10]=np.clip(2*(goalq-q),-.1,.1)
  if k>190:
   c=xy(cu);cross=(c-b)-np.dot(c-b,d)*d
   a[:2]=np.clip(.025*d+.7*cross,-.06,.06)
  s,r,t,tr,i=e.step(a)
  if k%25==0 or t:
   print('DIR',direction,'STEP',k,'B',np.round(b,3),'C',np.round(xy(cu),3),'Q',np.round(q[[1,3,5]],2),'R',r,'T',t,flush=True)
  if t:break
 e.close()
with ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(run,[(0,-1),(1,0),(-1,0),(0,1)]))
