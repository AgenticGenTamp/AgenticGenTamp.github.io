from env_client import make_env
from approach import GeneratedApproach
import numpy as np
from scipy.spatial.transform import Rotation

def g(s,n,f):return float(s.get(s.get_object_from_name(n),f))
e=make_env();s,i=e.reset(seed=0,options={'object_count':0});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
# Stop with blocker held, lifted and withdrawn.
while True:
 s,*_=e.step(p.get_action(s))
 if p.stage==3 and p.at(s,p.target(p.blocker+.36*p.out)):break
# Roll for clearance and lower shoulder outside the pen.
q=p.Q.copy();q[4]=p.w(q[4]+1.4);q[6]=p.w(q[6]+.4)
for _ in range(15):
 a=p.motion(s,p.robot(s))
 for j in range(7):
  d=q[j]-g(s,'robot','joint_'+str(j+1));d=p.w(d) if j in (4,6) else d;a[3+j]=np.clip(d,-.2,.2)
 s,*_=e.step(a)
tfp=np.array([g(s,'robot','grasp_tf_'+x) for x in 'xyz'])
tfr=Rotation.from_quat([g(s,'robot','grasp_tf_q'+x) for x in 'xyzw'])
bp=np.array([g(s,'blocker','pose_'+x) for x in 'xyz']);br=Rotation.from_quat([g(s,'blocker','pose_q'+x) for x in 'xyzw'])
tr=br*tfr.inv();tp=bp-tr.apply(tfp);base=np.r_[p.robot(s),0.]
print('base',base,'yaw',g(s,'robot','base_rot'),'q',q.tolist())
print('tool',tp.tolist(),tr.as_quat().tolist(),'local', (Rotation.from_euler('z',-g(s,'robot','base_rot')).apply(tp-base)).tolist(),'tfp',tfp.tolist())
e.close()
