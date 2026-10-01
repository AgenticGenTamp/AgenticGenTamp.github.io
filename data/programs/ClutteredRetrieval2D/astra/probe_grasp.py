from env_client import make_env
import numpy as np
import math

def position(s,name):
 o=s.get_object_from_name(name)
 return np.array([s.get(o,f) for f in ['x','y','theta']])
def center(s,name):
 p=position(s,name);o=s.get_object_from_name(name);wh=np.array([s.get(o,'width'),s.get(o,'height')])/2;c=math.cos(p[2]);q=math.sin(p[2]);return p[:2]+np.array([[c,-q],[q,c]])@wh

def go(e,s,goal,theta,arm=.1,vac=0,limit=100):
 for k in range(limit):
  p=position(s,'robot'); r=s.get_object_from_name('robot')
  d=np.r_[np.array(goal)-p[:2],(theta-p[2]+math.pi)%(2*math.pi)-math.pi,arm-s.get(r,'arm_joint'),vac]
  if max(abs(d[:4]))<1e-5: return s
  a=np.clip(d,e.action_space.low,e.action_space.high).astype(np.float32)
  q=s;s,_,term,_,_=e.step(a)
  if np.linalg.norm(position(q,'robot')-position(s,'robot'))<1e-6 and abs(a[3])<1e-6:
   print('blocked',position(s,'robot'),a);return s
 return s

if __name__=='__main__':
 from approach import GeneratedApproach
 e=make_env();s,i=e.reset(seed=2);policy=GeneratedApproach(e.action_space,e.observation_space,{});policy.reset(s,i)
 for k in range(200):
  a=policy.get_action(s);pre=s;s,_,term,_,_=e.step(a)
  if policy.phase=='grasp':
   print('GRASP',k,'chosen',policy.chosen,'q',position(pre,'robot'),'object',position(pre,policy.chosen),'center',center(pre,policy.chosen),'action',a,'moved',center(s,policy.chosen)-center(pre,policy.chosen))
   if policy.creeps>6:
    th=position(s,'robot')[2]
    for j in range(8):
     pre=s;a=np.array([-.003*math.cos(th),-.003*math.sin(th),0,0,1],dtype=np.float32);s,_,_,_,_=e.step(a);print('BACK',j,'q',position(s,'robot'),'moved',center(s,policy.chosen)-center(pre,policy.chosen))
    for n in s.get_object_names():
     if n!='robot':
      o=s.get_object_from_name(n);print('OBJECT',n,position(s,n),'center',center(s,n),'size',[s.get(o,f) for f in ['width','height']])
    for a in [[0,0,0,-.05,1],[0,0,.01,0,1],[0,0,-.01,0,1]]+[[x,y,0,0,1] for x,y in [(0.01,0),(-.01,0),(0,.01),(0,-.01),(.01,.01),(-.01,-.01),(.01,-.01),(-.01,.01)]]:
     pre=s;s,_,_,_,_=e.step(np.array(a,dtype=np.float32));print('TRY',a,'q',position(s,'robot'),'moved',[(n,(position(s,n)-position(pre,n)).round(5).tolist()) for n in s.get_object_names() if n!='robot' and np.linalg.norm(position(s,n)-position(pre,n))>1e-6])
    break
  if term:print('SUCCESS',k);break
 e.close()
