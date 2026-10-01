import numpy as np
from env_client import make_env

env=make_env(); o,_=env.reset(seed=0)
def go(a):
 global o
 o,r,t,tr,_=env.step(np.asarray(a,np.float32)); return r
def act(x=0,y=0,yaw=0,g=0):
 a=np.zeros(11); a[0]=x;a[1]=y;a[2]=yaw;a[10]=g;return a

# Wide collision-free route to the negative-y long face, rightmost drawer.
for _ in range(15): go(act(y=-.1))
print("south",o[125:128].round(3))
for _ in range(5): go(act(x=-.1))
print("west",o[125:128].round(3))
for _ in range(18): go(act(yaw=-.1))
print("north-facing",o[125:128].round(3))
for k in range(5):
 go(act(y=.1)); print("approach",k,o[125:128].round(3),o[103:109].round(4))
for k in range(2):
 go(act(g=1));print("close",k,o[135],o[103:109].round(4))
for k in range(5):
 r=go(act(y=-.1,g=1));print("pull",k,o[125:128].round(3),o[103:109].round(4),r)
env.close()
