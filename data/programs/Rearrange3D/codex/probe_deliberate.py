import numpy as np
from env_client import make_env

env=make_env(); s,_=env.reset(seed=0); initial=s.copy()
target_y=float(s[17])
def do(a,n):
 global s
 for _ in range(n): s,r,t,tr,i=env.step(np.asarray(a,np.float32))

# Translate laterally behind the counter, then approach it.
for _ in range(12):
 a=np.zeros(11); a[1]=np.clip((target_y-s[94])*.2,-.1,.1); do(a,1)
print('lateral',np.round(s[93:96],3))
for _ in range(8):
 a=np.zeros(11); a[0]=.1; do(a,1)
print('front',np.round(s[93:96],3))
# Unfold elbow (joint4 is action index 6), checking contact each step.
for k in range(60):
 a=np.zeros(11); a[6]=.1
 s,r,t,tr,i=env.step(np.asarray(a,np.float32))
 d=s[[0,1,2,16,17,18,32,33,34]]-initial[[0,1,2,16,17,18,32,33,34]]
 if k%5==4 or np.max(abs(d))>.003:
  print(k, 'q4',round(float(s[99]),3),'obj',np.round(d,3),'r',r)
 if np.max(abs(d))>.003: break
env.close()
