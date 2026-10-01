from env_client import make_env
import numpy as np,math
E=make_env();s,_=E.reset(seed=0)
u=np.array([math.cos(s[11]),math.sin(s[11])]); p=s[9:11]-s[18]*u
base=p-.2*np.array([1.,0.])
for t in range(80):
 dtheta=(0-s[2]+math.pi)%(2*math.pi)-math.pi
 a=np.clip([*(base-s[:2]),dtheta,.2-s[4],0],[-.05,-.05,-math.pi/16,-.1,0],[.05,.05,math.pi/16,.1,1])
 s,r,d,tr,i=E.step(a.astype(np.float32))
 if t%10==0:print(t,s[:12].round(4))
print('endpoint',p,'base target',base)
for t in range(12):
 a=np.array([.005,0,0,0,1],dtype=np.float32)
 s,*_=E.step(a);print('contact',t,s[:12].round(4))
for t in range(6):
 s,*_=E.step(np.array([.05,0,0,0,1],dtype=np.float32)); print('move',t,s[:12].round(4))
E.close()
