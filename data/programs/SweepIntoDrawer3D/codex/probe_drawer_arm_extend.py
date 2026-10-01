import numpy as np
from env_client import make_env

env=make_env();o,_=env.reset(seed=0)
def s(a):
 global o
 o,r,t,tr,_=env.step(np.asarray(a,np.float32));return r
def a(x=0,y=0,yaw=0,j4=0,g=0):
 z=np.zeros(11);z[0]=x;z[1]=y;z[2]=yaw;z[6]=j4;z[10]=g;return z
for _ in range(15):s(a(y=-.1))
for _ in range(5):s(a(x=-.1))
for _ in range(18):s(a(yaw=-.1))
for _ in range(5):s(a(y=.1))
for k in range(9):
 r=s(a(j4=.1)); print("extend",k,"base",o[125:128].round(3),"q",o[128:135].round(2),"draw",o[103:109].round(4),"w",o[147:150].round(3),r)
print("STATE_JSON", repr(o.tolist()))
for _ in range(2):s(a(g=1))
for k in range(7):
 r=s(a(y=-.1,g=1));print("pull",k,o[125:128].round(3),o[103:109].round(4),r)
env.close()
