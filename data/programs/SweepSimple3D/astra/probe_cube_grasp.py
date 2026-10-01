from env_client import make_env
import numpy as np

def v(s,n,fs):
 o=s.get_object_from_name(n);return np.array([float(s.get(o,f)) for f in fs])
for gap in [.55,.65,.75,.85]:
 for dx in [-.06,0,.06]:
  e=make_env();s,i=e.reset(seed=1)
  c0=v(s,'cube_0',['x','y','z']);q=np.array([0,2.4,np.pi,.3,0,-.873,np.pi/2])
  def move(target,grip,steps):
   global s
   peak=0
   for k in range(steps):
    a=np.zeros(11,dtype=np.float32)
    actual=v(s,'robot',['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(j) for j in range(1,8)])
    a[:3]=np.clip(np.array(target)-actual[:3],-.07,.07);a[3:10]=np.clip(q-actual[3:],-.1,.1);a[10]=grip
    s,r,t,tr,i=e.step(a)
    peak=max(peak,v(s,'cube_0',['z'])[0])
   return v(s,'cube_0',['x','y','z']),peak
  base=[c0[0]+dx,c0[1]+gap,-np.pi/2]
  before,_=move(base,0,150)
  closed,_=move(base,1,10)
  q[1]=1.2
  after,peak=move(base,1,40)
  print('TRIAL',gap,dx,'initial',np.round(c0,3),'before',np.round(before,3),'closed',np.round(closed,3),'after',np.round(after,3),'peak',round(peak,4),flush=True)
  e.close()
