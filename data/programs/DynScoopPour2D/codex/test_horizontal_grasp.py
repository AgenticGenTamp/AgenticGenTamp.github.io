import sys,math
import numpy as np
from env_client import make_env
xoff=float(sys.argv[1]); y=float(sys.argv[2]); seed=int(sys.argv[3])
e=make_env();s,i=e.reset(seed=seed);g=lambda n,f:float(s.get(s.get_object_from_name(n),f));hx=g('hook','x')
def go(x,y,th,gap,n):
 global s
 for _ in range(n):
  er=(th-g('robot','theta')+math.pi)%(2*math.pi)-math.pi
  s,*_=e.step(np.array([np.clip(x-g('robot','x'),-.03,.03),np.clip(y-g('robot','y'),-.03,.03),np.clip(er,-.098,.098),0,gap]))
go(2.7,2.1,0,.015,80);go(hx+xoff,1.2,0,.015,50);go(hx+xoff,y,0,.015,50);go(hx+xoff,y,0,-.015,20)
print(xoff,y,'held',g('hook','held'),'r',g('robot','x'),g('robot','y'),'h',g('hook','x'),g('hook','y'))
e.close()
