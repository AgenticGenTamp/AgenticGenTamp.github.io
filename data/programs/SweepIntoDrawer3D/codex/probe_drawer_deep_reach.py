import numpy as np
from env_client import make_env

env=make_env();o,_=env.reset(seed=0)
def go(x=0,y=0,yaw=0,j4=0,g=0):
 global o
 a=np.zeros(11,np.float32);a[[0,1,2,6,10]]=[x,y,yaw,j4,g]
 o,r,t,tr,_=env.step(a);return r
for _ in range(15):go(y=-.1)
for _ in range(5):go(x=-.1)
for _ in range(18):go(yaw=-.1)
for _ in range(5):go(y=.1)
for k in range(36):
 r=go(j4=.1,g=1 if k>=24 else 0)
 if k%3==2:print("reach",k+1,"q4",round(float(o[131]),3),"draw",o[103:109].round(4),r)
for k in range(10):
 r=go(y=-.1,g=1);print("pull",k,o[125:128].round(3),o[103:109].round(4),r)
env.close()
