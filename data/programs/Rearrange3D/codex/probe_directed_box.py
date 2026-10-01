from env_client import make_env
import numpy as np

env=make_env(); s,_=env.reset(seed=0); ix=np.array([0,1,2,16,17,18,32,33,34])
def run(a,n,label):
 global s
 for k in range(n):
  old=s;s,r,t,tr,_=env.step(np.array(a,np.float32));dd=s[ix]-old[ix]
  if np.max(np.abs(dd))>.002 or r!=-1:print(label,k,'r',r,'b',np.round(s[93:96],2),'q',np.round(s[[97,99]],2),'dd',np.round(dd.reshape(3,3),3),flush=True)
# Align just above the box while folded.
for k in range(35):
 a=np.zeros(11,np.float32); goal=s[17]+.28
 a[1]=np.clip((goal-s[94])*.5,-.1,.1); run(a,1,'above')
 if abs(goal-s[94])<.025: break
for k in range(70):
 a=np.zeros(11,np.float32)
 if abs(s[97]-.68)>.04:a[4]=.1*np.sign(.68-s[97])
 if abs(s[99]+1.70)>.04:a[6]=.1*np.sign(-1.70-s[99])
 if not np.any(a):break
 run(a,1,'cfg')
print('ready',np.round(s[[93,94,97,99]],2))
# Gripper zero leaves two fingers as a broad counter-height pusher.
run([0,-.1,0,0,0,0,0,0,0,0,0],40,'pushbox')
env.close()
