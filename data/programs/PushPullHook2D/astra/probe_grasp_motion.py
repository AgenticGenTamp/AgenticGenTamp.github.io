from env_client import make_env
import numpy as np
np.set_printoptions(precision=4,suppress=True)
def step(e,a):return e.step(np.array(a,dtype=np.float32))[0]
def grasp(e):
 s,_=e.reset(seed=0);u=np.array([np.cos(s[11]),np.sin(s[11])]);p=s[9:11]-1.22*u
 for j in range(100):
  th=np.arctan2(np.sin(np.pi-s[2]),np.cos(np.pi-s[2]));d=p+np.array([.4,0])-s[:2]
  s=step(e,[*np.clip(d,-.05,.05),np.clip(th,-.196,.196),-.1,0])
  if np.max(np.abs(d))<1e-5 and abs(th)<1e-5:break
 for j in range(46):s=step(e,[-.005,0,0,0,1])
 return s
for rotation in [-.1,.1]:
 e=make_env();s=grasp(e)
 for j in range(65):
  old=s;s=step(e,[0,0,rotation,0,1])
  if j%5==0 or np.max(abs(s-old))<1e-6: print('rotation',rotation,'j',j,'base',s[:5],'hook',s[9:12])
  if np.max(abs(s-old))<1e-6:break
 e.close()
# Align shaft upright, put far corner just below button and push up.
e=make_env();s=grasp(e)
for j in range(40):
 d=np.arctan2(np.sin(np.pi/2-s[11]),np.cos(np.pi/2-s[11]))
 s=step(e,[0,0,np.clip(d,-.05,.05),0,1])
 if abs(d)<1e-5:break
print('upright',s.tolist())
for j in range(100):
 d=s[20:22]+np.array([0,-.1])-s[9:11]
 old=s;s=step(e,[*np.clip(d,-.025,.025),0,0,1])
 if j%10==0:print('position',j,'base',s[:2],'hook',s[9:12],'button',s[20:22])
 if np.linalg.norm(s[:2]-old[:2])<1e-6:break
for j in range(40):
 old=s;s=step(e,[0,.01,0,0,1])
 if j%5==0:print('pushup',j,'base',s[:2],'hook',s[9:12],'button',s[20:22])
e.close()
# Contact propagation directions at crossbar
e=make_env();s=grasp(e)
for j in range(40):
 d=np.arctan2(np.sin(np.pi/2-s[11]),np.cos(np.pi/2-s[11]));s=step(e,[0,0,np.clip(d,-.05,.05),0,1])
 if abs(d)<1e-5:break
for j in range(100):
 d=s[20:22]+np.array([0.,-.1])-s[9:11]
 s=step(e,[*np.clip(d,-.025,.025),0,0,1])
 if np.linalg.norm(d)<1e-5:break
for a in [[0,.03,0,0,1],[.02,0,0,0,1],[0,-.02,0,0,1],[0,.02,0,0,1],[-.02,.005,0,0,1],[-.02,0,0,0,1],[0,-.02,0,0,1]]:
 old=s;s=step(e,a);print('contactprop',a,'hookdelta',s[9:11]-old[9:11],'buttondelta',s[20:22]-old[20:22], 'gap',s[20:22]-s[9:11])
e.close()
