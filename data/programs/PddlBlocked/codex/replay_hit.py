from env_client import make_env
import numpy as np
def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def move(e,s,tx,ty,q):
 for _ in range(20):
  a=np.zeros(11,np.float32);a[0]=np.clip(tx-g(s,'robot','base_x'),-.2,.2);a[1]=np.clip(ty-g(s,'robot','base_y'),-.2,.2);a[10]=1
  for j in range(7):
   d=q[j]-g(s,'robot','joint_'+str(j+1));d=(d+np.pi)%(2*np.pi)-np.pi if j in (4,6) else d;a[3+j]=np.clip(d,-.2,.2)
  s,*_=e.step(a)
 return s
Q=np.array([0.16752,0.21063,1.4,-0.24177,-2.99431,-0.83787,-3.14159])
e=make_env();s,_=e.reset(seed=27)
print('poses',[(n,g(s,n,'pose_x'),g(s,n,'pose_y')) for n in s.get_object_names() if n.startswith('green') or n=='blocker'])
s=move(e,s,3.496114254,-.214349046,Q)
a=np.zeros(11,np.float32);a[10]=-1;s,r,t,tr,i=e.step(a)
print('grasp',g(s,'robot','grasp_active'),'base',g(s,'robot','base_x'),g(s,'robot','base_y'),'q',[g(s,'robot','joint_'+str(k)) for k in range(1,8)])
print('active',[(n,g(s,n,'grasp_active')) for n in s.get_object_names() if n.startswith('green') or n=='blocker'])
print('tf',[g(s,'robot','grasp_tf_'+x) for x in ['x','y','z','qx','qy','qz','qw']])
print('blockquat',[g(s,'blocker','pose_q'+x) for x in ['x','y','z','w']])
e.close()
