import numpy as np
from scipy.spatial.transform import Rotation
from kinematics import ik, fk, Q0, DOWN

class GeneratedApproach:
 def __init__(self,action_space,observation_space,primitives):
  self.space=observation_space
  self.rt=observation_space.get_type('robot');self.bt=observation_space.get_type('block')
 def reset(self,state,info):
  self.robot=state.get_objects(self.rt)[0];self.queue=[];self.target=None;self.stage='select';self.prev=None;self.stuck=0;self.side=-1;self.done=set();self.steps=0
  self.plate=state.get_object_from_name('plate')
  self.slots=[np.array([x,y]) for x in [-.08,0,.08] for y in [-.08,0,.08]]
 def conf(self,s):
  return np.array([s.get(self.robot,k) for k in ['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]])
 def pos(self,s,o):return np.array([s.get(o,'pose_'+k) for k in 'xyz'])
 def yaw(self,s,o):return 2*np.arctan2(s.get(o,'pose_qz'),s.get(o,'pose_qw'))
 def add(self,q,grip=0):self.queue.append((np.array(q),grip))
 def solve(self,base,xyz,yaw,q0=Q0):
  br=Rotation.from_euler('z',base[2]).as_matrix()
  p=br.T@(np.array(xyz)-np.r_[base[:2],0.])
  rot=br.T@Rotation.from_euler('z',yaw).as_matrix()@DOWN
  return np.r_[base,ik(p,q0=q0,rot=rot)]
 def location(self,xy,side):
  yaw=0 if side==-1 else np.pi
  return np.array([min(xy[0]-.70,-.66) if side==-1 else max(xy[0]+.70,.66),xy[1]-.25 if side==-1 else xy[1]+.25,yaw])
 def route(self,cur,base):
  # Keep hand above obstacles while driving around either end of table.
  if cur[0]*base[0]<0:
   edge=1.15 if cur[1]+base[1]>0 else -1.15
   self.add(np.r_[cur[:3],Q0],1)
   a=cur[:3].copy();a[0]=-.85 if cur[0]<0 else .85;a[1]=edge;a[2]=np.sign(edge)*np.pi/2;self.add(np.r_[a,Q0],1)
   a[0]=base[0];a[2]=base[2];self.add(np.r_[a,Q0],1)
   a[1]=base[1];self.add(np.r_[a,Q0],1)
  else:pass
 def get_action(self,s):
  self.steps+=1;cur=self.conf(s);held=s.get(self.robot,'grasp_active')>.5
  if self.prev is not None and np.max(np.abs(cur-self.prev))<1e-6:self.stuck+=1
  else:self.stuck=0
  self.prev=cur.copy()
  for _ in range(15):
   if self.queue:
    goal,grip=self.queue[0];d=goal-cur
    for j in [2,7,9]:d[j]=(d[j]+np.pi)%(2*np.pi)-np.pi
    if np.max(abs(d))<.002:
     self.queue.pop(0)
     if grip>0 and not held and s.get(self.robot,'gripper_opening')>.5:continue
     if grip<0 and held:continue
     if grip:return np.r_[np.zeros(10),grip].astype(np.float32)
     continue
    return np.r_[d*min(1.,.2/max(np.max(abs(d)),1e-9)),grip].astype(np.float32)
   if self.stage=='select':
    blocks=[b for b in s.get_objects(self.bt) if b.name not in self.done]
    if not blocks:return np.zeros(11,dtype=np.float32)
    b=min(blocks,key=lambda b:abs(self.pos(s,b)[0]-cur[0])+.3*abs(self.pos(s,b)[1]-cur[1]))
    self.target=b;pos=self.pos(s,b);self.side=(-1 if cur[0]<0 else 1) if abs(pos[0])<.08 else (-1 if pos[0]<0 else 1)
    base=self.location(pos[:2],self.side);yaw=self.yaw(s,b)
    # Square symmetry avoids unnecessary wrist rotation.
    yaw=(yaw+np.pi/4)%(np.pi/2)-np.pi/4+base[2]
    self.route(cur,base)
    high=self.solve(base,[*pos[:2],1.04],yaw)
    low=self.solve(base,[*pos[:2],pos[2]+.038],yaw,q0=high[3:])
    self.add(high,1)
    mid=high
    for z in [1.0,.94,.88]:
     mid=self.solve(base,[*pos[:2],z],yaw,q0=mid[3:]);self.add(mid,1)
    low=self.solve(base,[*pos[:2],pos[2]+.038],yaw,q0=mid[3:])
    self.add(low,1);self.add(low,-1)
    self.stage='lift'
   elif self.stage=='lift':
    if not held:
     self.stage='select';self.add(np.r_[cur[:3],Q0],1);continue
    pos=self.pos(s,self.target)
    p,r=fk(cur[3:]);p[2]+=.20
    self.add(np.r_[cur[:3],ik(p,q0=cur[3:],rot=r)])
    self.stage='place'
   elif self.stage=='place':
    plate=self.pos(s,self.plate)
    slots=[v+plate[:2] for v in self.slots]
    blocks=[b for b in s.get_objects(self.bt) if b!=self.target]
    valid=[v for v in slots if all(np.max(abs(v-self.pos(s,b)[:2]))>.071 for b in blocks)]
    if not valid:valid=slots
    v=min(valid,key=lambda v:abs(v[0]-cur[0])+.1*abs(v[1]-cur[1]))
    base=self.location(v,self.side)
    # Align held block to table axes using the observed rigid grasp transform.
    tf=np.array([s.get(self.robot,'grasp_tf_'+k) for k in ['qx','qy','qz','qw']])
    inv_tf=Rotation.from_quat(tf).inv().as_matrix()
    current_rot=Rotation.from_euler('z',cur[2]).as_matrix()@fk(cur[3:])[1]
    candidates=[Rotation.from_euler('z',k*np.pi/2).as_matrix()@inv_tf for k in range(4)]
    desired=min(candidates,key=lambda r:np.linalg.norm(Rotation.from_matrix(current_rot.T@r).as_rotvec()))
    br=Rotation.from_euler('z',base[2]).as_matrix()
    offset=np.array([s.get(self.robot,'grasp_tf_'+k) for k in 'xyz'])
    xyz=np.r_[v,1.01]-desired@offset
    q=ik(br.T@(xyz-np.r_[base[:2],0]),q0=cur[3:],rot=br.T@desired)
    self.add(np.r_[base,q]);self.add(np.r_[base,q],1)
    self.stage='release'
   elif self.stage=='release':
    if not held:self.done.add(self.target.name);self.stage='select'
    else:self.stage='place'
  return np.zeros(11,dtype=np.float32)
