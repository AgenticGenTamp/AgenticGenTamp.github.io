from env_client import make_env
import numpy as np
e=make_env();s,_=e.reset(seed=0);r=s.get_object_from_name('robot')
def step(a):
 global s
 s,*_=e.step(np.array(a,np.float32))
def p():return np.array([s.get(r,f) for f in ['x','y','theta','arm_joint']])
for a in [[0,0,0,-.001,0],[0,0,0,-.05,0],[0,0,0,-.05,0],[0,0,0,.1,0],[0,0,0,.1,0],[0,0,0,.001,0]]:
 step(a);print('arm',a[3],p())
for axis,sgn in [(0,-1),(1,-1),(1,1)]:
 for i in range(100):
  a=np.zeros(5);a[axis]=sgn*.05;old=p();step(a)
  if np.linalg.norm(p()-old)<1e-7:print('boundary',axis,sgn,p());break
 for _ in range(2):
  a=np.zeros(5);a[axis]=-sgn*.05;step(a)
e.close()
