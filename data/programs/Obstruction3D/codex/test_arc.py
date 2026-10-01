from env_client import make_env
import numpy as np, math

def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def center(a,b,r=.656):
 m=((a[0]+b[0])/2,(a[1]+b[1])/2); dx=b[0]-a[0];dy=b[1]-a[1];d=math.hypot(dx,dy);h=math.sqrt(r*r-d*d/4)
 cs=[(m[0]-dy/d*h,m[1]+dx/d*h),(m[0]+dy/d*h,m[1]-dx/d*h)]
 return min(cs,key=lambda x:x[0])
e=make_env();s,_=e.reset(seed=4);n='obstruction0';src=(g(s,n,'pose_x'),g(s,n,'pose_y'));dst=(.2,0);c=center(src,dst);qs=-math.atan2(src[1]-c[1],src[0]-c[0]);qd=-math.atan2(dst[1]-c[1],dst[0]-c[0]);print(src,dst,c,qs,qd)
for k in range(20):
 a=np.zeros(11,np.float32)
 for idx,feat,goal in [(0,'pos_base_x',c[0]),(1,'pos_base_y',c[1]),(3,'joint_1',qs),(4,'joint_2',.65),(6,'joint_4',-1.5)]:a[idx]=np.clip(goal-g(s,'robot',feat),-.2,.2)
 a[10]=-1;s,*_=e.step(a)
 if g(s,'robot','grasp_active'):print('grasp',k,tuple(g(s,n,f) for f in ('pose_x','pose_y','pose_z')));break
a=np.zeros(11,np.float32);a[8]=.1;s,*_=e.step(a);print('lift',tuple(g(s,n,f) for f in ('pose_x','pose_y','pose_z')))
for k in range(30):
 a=np.zeros(11,np.float32);a[3]=np.clip(qd-g(s,'robot','joint_1'),-.2,.2);s,*_=e.step(a)
 if abs(qd-g(s,'robot','joint_1'))<.01:break
print('moved',k,tuple(g(s,n,f) for f in ('pose_x','pose_y','pose_z')),g(s,'robot','joint_1'))
a=np.zeros(11,np.float32);a[8]=-.1;s,*_=e.step(a);print('lower',tuple(g(s,n,f) for f in ('pose_x','pose_y','pose_z')))
a=np.zeros(11,np.float32);a[10]=1;s,*_=e.step(a);print('open',g(s,'robot','grasp_active'),tuple(g(s,n,f) for f in ('pose_x','pose_y','pose_z')))
e.close()
