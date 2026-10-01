import numpy as np
from env_client import make_env

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def p(s,n): return np.array([v(s,n,f) for f in 'xyz'])
def q(s): return np.array([v(s,'robot',f'pos_arm_joint{i}') for i in range(1,8)])
e=make_env();s,_=e.reset(seed=0);w0=p(s,'wiper_0')
for _ in range(3):
 a=np.zeros(11,np.float32);a[1]=-.1;s,_,_,_,_=e.step(a)
target=np.array([-3.5,-.04,2.84,-1.2,-.06,-.49,1.57])
for i in range(120):
 a=np.zeros(11,np.float32); a[3:10]=np.clip((target-q(s))*.5,-.1,.1); a[10]=0
 s,r,d,tr,_=e.step(a)
 if i%10==9 or np.linalg.norm(p(s,'wiper_0')-w0)>.002: print(i+1,'q',q(s).round(2),'wd',(p(s,'wiper_0')-w0).round(3),'r',r)
 if np.max(np.abs(target-q(s)))<.03:break
for i in range(5):
 a=np.zeros(11,np.float32);a[10]=1;s,r,d,tr,_=e.step(a)
print('closed?',v(s,'robot','pos_gripper'),'w',p(s,'wiper_0').round(3))
for goal in np.linspace(-3.5,2.5,13):
 for i in range(25):
  a=np.zeros(11,np.float32);a[3]=np.clip((goal-q(s)[0])*.5,-.1,.1);a[10]=1;s,r,d,tr,_=e.step(a)
  if np.linalg.norm(p(s,'wiper_0')-w0)>.002: print('CONTACT goal',goal,'i',i,'q1',q(s)[0],'wd',p(s,'wiper_0')-w0)
 print('scan',goal,'q1',round(q(s)[0],2),'wd',(p(s,'wiper_0')-w0).round(3))
e.close()
