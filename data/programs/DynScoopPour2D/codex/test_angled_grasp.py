import sys,math
import numpy as np
from env_client import make_env
y=float(sys.argv[1]); th=float(sys.argv[2]);e=make_env();s,i=e.reset(seed=14);g=lambda n,f:float(s.get(s.get_object_from_name(n),f))
def go(x,y,th,gap,n,arm=.08):
 global s
 for _ in range(n):
  er=(th-g('robot','theta')+math.pi)%(2*math.pi)-math.pi
  s,*_=e.step(np.array([np.clip(x-g('robot','x'),-.03,.03),np.clip(y-g('robot','y'),-.03,.03),np.clip(er,-.098,.098),arm,gap]))
go(3.15,2.2,th,.015,80);go(3.15,y,th,.015,60);go(3.28,y,th,.015,10);go(3.28,y,th,-.015,20)
print(y,th,g('hook','held'),g('hook','x'),g('hook','y'),g('robot','x'),g('robot','y'))
e.close()
