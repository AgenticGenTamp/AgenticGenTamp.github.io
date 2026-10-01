"""Focused close-and-tug around the visually aligned Gen3 configuration."""
import itertools, math, sys
import numpy as np
from env_client import make_env
from probe_fk_model import fk

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def xyz(s,n): return np.array([v(s,n,f) for f in 'xyz'])
def q(s): return np.array([v(s,'robot',f'pos_arm_joint{i}') for i in range(1,8)])
def base(s): return np.array([v(s,'robot','pos_base_x'),v(s,'robot','pos_base_y')])
def ae(a,b): return (a-b+math.pi)%(2*math.pi)-math.pi

e=make_env(); yaw=-1.475
mode=int(sys.argv[1]) if len(sys.argv)>1 else 0
if mode==0: candidates=itertools.product((.9,1.1,1.3,1.5,1.7),(-2.57,))
elif mode==1: candidates=itertools.product((-.16,-.08,0,.08,.16),(-.16,-.08,0,.08,.16))
else: candidates=[(-.16,-.16)]
for vals in candidates:
 s,_=e.reset(seed=0,options={'object_count':1});w0=xyz(s,'wiper_0');
 if mode==0: q2,q4=vals; fwd=lat=0
 else: fwd,lat=vals;q2=.8;q4=-2.57
 R=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
 qt=np.array([0,q2,2.37,q4,.02,-.8,1.57])
 # Counteract the approximate FK's horizontal shift so only wrist height changes.
 dp=fk(qt)[:2,3]-fk(np.array([0,.32,2.37,-2.57,.02,-.8,1.57]))[:2,3]
 bt=w0[:2]-R@(np.array([.400+fwd,.360+lat])+dp)
 maxdist=0
 for k in range(45):
  a=np.zeros(11,np.float32);a[:2]=np.clip(.8*(bt-base(s)),-.1,.1)
  a[2]=np.clip(.8*ae(yaw,v(s,'robot','pos_base_rot')),-.1,.1)
  a[3:10]=np.clip(.7*(qt-q(s)),-.1,.1);a[10]=1
  s,*_=e.step(a);maxdist=max(maxdist,np.linalg.norm(xyz(s,'wiper_0')-w0))
  if mode==2 and maxdist>.002: break
 pre=xyz(s,'wiper_0').copy();
 for k in range(8):
  a=np.zeros(11,np.float32);a[3:10]=np.clip(.5*(qt-q(s)),-.1,.1);a[10]=0;s,*_=e.step(a)
 # Tug base back toward its initial northern location.
 for k in range(8):
  a=np.zeros(11,np.float32);a[1]=.07;a[3:10]=np.clip(.5*(qt-q(s)),-.1,.1);a[10]=0;s,*_=e.step(a)
 follow=np.linalg.norm(xyz(s,'wiper_0')-pre)
 print('cand',vals,'q',q(s).round(3),'g',round(v(s,'robot','pos_gripper'),3),
       'contact',round(maxdist,4),'follow',round(follow,4),'wd',(xyz(s,'wiper_0')-w0).round(3),flush=True)
 if follow>.03: print('SUCCESS',vals,flush=True);break
e.close()
