import numpy as np
from env_client import make_env
np.set_printoptions(precision=8,suppress=True)
def rot(a):return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
e=make_env()
for mode in ['bar_top','bar_bottom','stem_bottom','stem_left']:
 o,_=e.reset(seed=0);c=o[:2].copy();R=rot(o[2]);lh=o[13];lv=o[14];w=o[12]
 def go(pt):
  global o
  for _ in range(150):
   a=np.clip(pt-o[16:18],-.049,.049)
   if np.linalg.norm(a)<1e-5:break
   o,*_=e.step(a)
 rel=R.T@(o[16:18]-c);angle=np.arctan2(rel[1],rel[0]);rad=max(1.8,np.linalg.norm(rel));go(c+R@np.array([np.cos(angle),np.sin(angle)])*rad)
 target={'bar_top':np.array([.25,1.]),'bar_bottom':np.array([.25,-.6]),'stem_bottom':np.array([0.,-1.7]),'stem_left':np.array([-1.,-.5])}[mode]
 ta=np.arctan2(target[1],target[0]);delta=(ta-angle+np.pi)%(2*np.pi)-np.pi
 for t in np.linspace(angle,angle+delta,30):go(c+R@np.array([np.cos(t),np.sin(t)])*rad)
 go(c+R@target)
 a={'bar_top':np.array([0,-.01]),'bar_bottom':np.array([0,.01]),'stem_bottom':np.array([0,.01]),'stem_left':np.array([.01,0])}[mode]
 print(mode,'dims',w,lh,lv,'navdelta',o[:2]-c)
 for k in range(150):
  n,*_=e.step(R@a);dd=n[:3]-o[:3]
  if np.linalg.norm(dd)>1e-7:
   print('firstcontact before local robot',R.T@(o[16:18]-o[:2]),'after',R.T@(n[16:18]-o[:2]),'delta',R.T@dd[:2],dd[2]);break
  o=n
e.close()
