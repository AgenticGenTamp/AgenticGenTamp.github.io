from env_client import make_env
import numpy as np

def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def drive(e,s,tx,ty):
  for k in range(10):
    a=np.zeros(11,np.float32); a[0]=np.clip(tx-g(s,'robot','base_x'),-.2,.2);a[1]=np.clip(ty-g(s,'robot','base_y'),-.2,.2);a[10]=1
    s,*_=e.step(a)
  return s
def goto(e,s,q):
  for k in range(18):
    a=np.zeros(11,np.float32)
    for i in range(7):
      d=q[i]-g(s,'robot','joint_'+str(i+1)); d=(d+np.pi)%(2*np.pi)-np.pi if i in (4,6) else d
      a[3+i]=np.clip(d,-.2,.2)
    a[10]=1
    s,*_=e.step(a)
    if max(abs(a[3:10]))<.01:break
  return s

rng=np.random.default_rng(5);e=make_env()
lo=np.array([-2.28,-.52,-np.pi,-2.32,-np.pi,-2.18,-np.pi]); hi=np.array([.71,1.39,np.pi,0,np.pi,0,np.pi])
for ep in range(300):
 s,_=e.reset(seed=ep)
 b=g(s,'blocker','pose_y')
 # robot left arm is on +y side: cover several lateral offsets
 bx=3.35+rng.uniform(-.15,.15);by=b+rng.uniform(-.6,.2)
 s=drive(e,s,bx,by)
 for trial in range(45):
  if rng.random()<.55:
   init=np.array([.3928,.3333,0,-1.5224,2.7217,-1.2195,-2.9891])
   q=np.clip(init+rng.normal(0,[.7,.45,1,.55,1,.55,1]),lo,hi)
  else:q=rng.uniform(lo,hi)
  s=goto(e,s,q)
  a=np.zeros(11,np.float32);a[10]=-1;s,*_=e.step(a)
  if g(s,'robot','grasp_active'):
   print('HIT',ep,trial,'base',g(s,'robot','base_x'),g(s,'robot','base_y'),'blocker',b,'q',[round(g(s,'robot','joint_'+str(i+1)),5) for i in range(7)],'tf',[round(g(s,'robot','grasp_tf_'+f),4) for f in ['x','y','z']]);raise SystemExit
e.close();print('none')
