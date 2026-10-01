from env_client import make_env
import numpy as np

def run(seed, waypoints):
 e=make_env();s,_=e.reset(seed=seed);r=s.get_object_from_name('rover0')
 def xy():return np.array([s.get(r,'x'),s.get(r,'y')])
 for target in waypoints:
  target=np.array(target)
  for i in range(100):
   p=xy();d=np.clip(target-p,-.2,.2);a=np.zeros(8);a[:2]=d
   s,_,_,_,_=e.step(a)
   if np.linalg.norm(xy()-target)<1e-5 or np.linalg.norm(xy()-p)<1e-7:break
  print('target',target,'reached',xy(),flush=True)
 e.close()

run(0,[(.4,-1.75),(.28,-1.75),(.275,-1.75),(.27,-1.75),(.4,-1.75),(.4,-2.25),(.2,-2.25),(0,-2.25),(-.2,-2.25)])
run(0,[(1,1.5),(.3,1.5),(.28,1.5),(.2,1.5),(0,1.5),(-.3,1.5),(1,1.5),(1,2.25),(.8,2.25),(.3,2.25),(0,2.25),(-.3,2.25)])
