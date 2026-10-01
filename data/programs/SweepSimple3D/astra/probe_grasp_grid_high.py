from env_client import make_env
import numpy as np

def v(s,n,fs):
 o=s.get_object_from_name(n);return np.array([float(s.get(o,f)) for f in fs])
for q2 in [1.7,2.0,2.3]:
 for q4 in [-1.5,-1.0]:
  e=make_env();s,i=e.reset(seed=1)
  w0=v(s,'wiper_0',['x','y','z']);q=np.array([0,q2,np.pi,q4,0,-.87266,np.pi/2])
  # set arm while positioning behind wiper, then approach increments.
  def move(target,grip,steps=20):
   global s
   for k in range(steps):
    a=np.zeros(11,dtype=np.float32)
    actual=v(s,'robot',['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(j) for j in range(1,8)])
    a[:3]=np.clip(np.array(target)-actual[:3],-.07,.07);a[3:10]=np.clip(q-actual[3:],-.1,.1);a[10]=grip
    s,r,t,tr,i=e.step(a)
   return v(s,'wiper_0',['x','y','z'])
  move([w0[0],w0[1]+.75,-np.pi/2],0,40)
  for gap in [.75,.55,.35]:
   before=move([w0[0],w0[1]+gap,-np.pi/2],0,8)
   closed=move([w0[0],w0[1]+gap,-np.pi/2],1,4)
   after=move([w0[0],w0[1]+gap+.2,-np.pi/2],1,8)
   print('TRIAL',q2,q4,gap,'initial',np.round(w0,3),'before',np.round(before,3),'closed',np.round(closed,3),'after',np.round(after,3),'delta',np.round(after-closed,3),flush=True)
  e.close()
