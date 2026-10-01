import numpy as np
from env_client import make_env

env=make_env();o,_=env.reset(seed=0)
def go(x=0,y=0,yaw=0,j2=0,j4=0,g=0):
 global o
 a=np.zeros(11,np.float32);a[[0,1,2,4,6,10]]=[x,y,yaw,j2,j4,g]
 o,r,t,tr,_=env.step(a);return r
for _ in range(15):go(y=-.1)
for _ in range(5):go(x=-.1)
for _ in range(18):go(yaw=-.1)
for _ in range(5):go(y=.1)
# Reach downward and toward the upper-right drawer handle, closing late.
for k in range(14):
 r=go(j2=-.1,j4=.1,g=1 if k>=8 else 0)
 print("hook",k,"q",o[128:135].round(2),"draw",o[103:109].round(4),r)
for k in range(7):
 r=go(y=-.1,g=1);print("pull",k,o[125:128].round(3),o[103:109].round(4),r)
env.close()
