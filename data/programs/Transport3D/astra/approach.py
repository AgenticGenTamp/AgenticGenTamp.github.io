"""Object-centric transport policy with calibrated kinematics and reactive grasp retries."""
import numpy as np
from scipy.optimize import least_squares
from kinova import fk

class GeneratedApproach:
 def __init__(self,action_space,observation_space,primitives):
  self.action_space=action_space; self.observation_space=observation_space
  self.ct=observation_space.get_type('Kinematic3DCuboid')
  self.rt=observation_space.get_type('Kinematic3DRobot')
  self.cache={}
 def reset(self,state,info):
  self.count=sum(1 for o in state.get_objects(self.ct) if o.name!='table' and state.get(o,'half_extent_x')<.06)
  self.phase='select';self.target=None;self.finished=set();self.slot=0;self.failed=0;self.tick=0;self.last=None;self.stuck=0;self.path=[]
 def xyz(self,s,o):return np.array([s.get(o,'pose_'+a) for a in 'xyz'])
 def ext(self,s,o):return np.array([s.get(o,'half_extent_'+a) for a in 'xyz'])
 def q_for(self,z,x=.4):
  key=(round(z,5),round(x,5))
  if key not in self.cache:
   def fun(v):
    T=fk([0,v[0],-np.pi,v[1],0,v[2],np.pi/2]);return np.r_[T[[0,2],3]-[x,z-.3948],(T[:3,2]-[0,0,-1])*.3]
   sol=least_squares(fun,[1.,-2.,-.2],bounds=([-2.4,-2.65,-2.22],[2.4,2.65,2.22]),max_nfev=45,ftol=1e-7,gtol=1e-7)
   self.cache[key]=np.array([0,sol.x[0],-np.pi,sol.x[1],0,sol.x[2],np.pi/2])
  return self.cache[key]
 def wrap(self,v):return (v+np.pi)%(2*np.pi)-np.pi
 def offset(self,yaw):
  c=np.cos(yaw);t=np.sin(yaw);return np.array([c*.5199-t*.001356,t*.5199+c*.001356])
 def blocked(self,p,rects):return any(np.all(p>lo)&np.all(p<hi) for lo,hi in rects)
 def segment(self,a,b,rects):
  for lo,hi in rects:
   d=b-a; t0=0.;t1=1.
   for j in range(2):
    if abs(d[j])<1e-10:
     if a[j]<=lo[j] or a[j]>=hi[j]:t1=-1;break
    else:
     v=sorted([(lo[j]-a[j])/d[j],(hi[j]-a[j])/d[j]])
     t0=max(t0,v[0]);t1=min(t1,v[1])
   if t0<t1 and t1>0 and t0<1:return False
  return True
 def route(self,a,b,rects):
  rects=[(lo,hi) for lo,hi in rects if not self.blocked(a,[(lo,hi)]) and not self.blocked(b,[(lo,hi)])]
  if self.segment(a,b,rects):return [b]
  nodes=[a,b]
  for lo,hi in rects:
   for x in [lo[0]-.015,hi[0]+.015]:
    for y in [lo[1]-.015,hi[1]+.015]:
     p=np.array([x,y])
     if not self.blocked(p,rects):nodes.append(p)
  dist=[float('inf')]*len(nodes);dist[0]=0;prev={};seen=set()
  while len(seen)<len(nodes):
   u=min((i for i in range(len(nodes)) if i not in seen),key=lambda i:dist[i])
   if u==1 or not np.isfinite(dist[u]):break
   seen.add(u)
   for v in range(len(nodes)):
    if v in seen:continue
    nd=dist[u]+np.linalg.norm(nodes[v]-nodes[u])
    if nd<dist[v] and self.segment(nodes[u],nodes[v],rects):dist[v]=nd;prev[v]=u
  if 1 not in prev:return [b]
  path=[];v=1
  while v: path.append(nodes[v]);v=prev[v]
  return path[::-1]
 def get_action(self,s):
  self.tick+=1;r=next(iter(s.get_objects(self.rt))); objs=list(s.get_objects(self.ct));table=next(o for o in objs if o.name=='table')
  base=np.array([s.get(r,'pos_base_x'),s.get(r,'pos_base_y')]);yaw=s.get(r,'pos_base_rot');q=np.array([s.get(r,'joint_'+str(i)) for i in range(1,8)])
  tab=self.xyz(s,table);te=self.ext(s,table);top=tab[2]+te[2]
  a=np.zeros(11,dtype=np.float32)
  rects=[]
  for o in objs:
   if o.name in self.finished or s.get(o,'grasp_active'):continue
   if o.name=='table' or self.ext(s,o)[0]>.06:
    p=self.xyz(s,o)[:2];e=self.ext(s,o)[:2]+.205;rects.append((p-e,p+e))
  cur=np.r_[base,yaw,q]
  self.stuck=self.stuck+1 if self.last is not None and np.max(abs(cur-self.last))<1e-5 else 0;self.last=cur
  for _ in range(8):
   if self.phase=='select':
    todo=[o for o in objs if o.name!='table' and o.name not in self.finished and not(abs(self.xyz(s,o)[0]-tab[0])<te[0] and abs(self.xyz(s,o)[1]-tab[1])<te[1] and self.xyz(s,o)[2]>=top+self.ext(s,o)[2]-.01)]
    if not todo:
     des=self.q_for(.85);delta=des-q
     a[3:10]=delta*min(1.,.12/max(np.max(abs(delta)),1e-8));a[10]=1
     return a
    # Place the larger box first, reserving the other half of the table for cubes.
    o=min(todo,key=lambda o:(-self.ext(s,o)[0],np.linalg.norm(self.xyz(s,o)[:2]-base)))
    self.target=o.name;self.pick=self.xyz(s,o);self.h=self.ext(s,o)[2]
    self.pick[2]+=self.h+(0.01 if self.h<.06 else 0)
    if self.h>.06:self.pick[1]-=self.ext(s,o)[1]
    choices=[]
    for ang in ([0,np.pi] if self.h>.06 else [0,np.pi/2,-np.pi/2,np.pi]):
     b=self.pick[:2]-self.offset(ang)
     score=np.linalg.norm(b-base)+.4*abs(ang)+(10 if self.blocked(b,rects) else 0)
     choices.append((score,ang,b))
    self.choices=sorted(choices,key=lambda v:v[0]);self.attempt=0
    _,self.pyaw,self.pbase=self.choices[0];self.box_yaw=self.pyaw;self.phase='raise_pick';self.grip=1;self.path=[];continue
   o=s.get_object_from_name(self.target)
   if self.phase in ['lower_pick','grasp'] and self.stuck>8:
    self.attempt+=1
    if self.h>.06:
     self.pick[1]=self.xyz(s,o)[1]+(1 if self.attempt%2 else -1)*self.ext(s,o)[1]
     self.pyaw=self.wrap(self.box_yaw+(np.pi if (self.attempt//2)%2 else 0))
     self.pbase=self.pick[:2]-self.offset(self.pyaw)
    else:
     _,self.pyaw,self.pbase=self.choices[self.attempt%len(self.choices)]
     if self.attempt%len(self.choices)==0:self.pick[2]+=.01
    self.phase='raise_pick';self.stuck=0;continue
   if self.phase in ['raise_pick','lift']:
    current_z=fk(q)[2,3]+.3948
    carry=.60 if self.count>12 and self.h<.06 else .72
    increment=.09 if self.count>12 and self.h<.06 else .06
    des=self.q_for(min(carry,current_z+increment))
    delta=des-q;a[3:10]=delta*min(1.,.2/max(np.max(abs(delta)),1e-8));a[10]=1 if self.phase=='raise_pick' else 0
    if np.max(abs(des-q))<.003 and abs(current_z-carry)<.003:
     if self.phase=='raise_pick':self.phase='travel_pick';self.path=self.route(base,self.pbase,rects);self.dyaw=self.pyaw
     else:
      if self.h>.06:self.place=np.array([tab[0],tab[1]+.20,top+self.h+.00001])
      else:
       if self.count<=6:
        ix=self.slot%2;iy=(self.slot//2)%3;layer=self.slot//6
        self.place=np.array([tab[0]-.09+ix*.18,tab[1]-.30+iy*.12,top+self.h+.00001+layer*.05001])
       else:
        cols=min(6,max(4,int(np.ceil(np.sqrt(self.count)))));rows=min(6,int(np.ceil(self.count/cols)))
        ix=self.slot%cols;iy=(self.slot//cols)%rows;layer=self.slot//(cols*rows)
        self.place=np.array([tab[0]+.13-ix*(.26/max(cols-1,1)),tab[1]-.32+iy*(.26/max(rows-1,1)),top+self.h+.00001+layer*.05001])
       self.slot+=1
      self.dyaw=0.;tf=np.array([s.get(r,"grasp_tf_"+v) for v in "xyz"]);self.placebase=self.place[:2]-np.array([tf[1],tf[0]])-self.offset(0);self.path=self.route(base,self.placebase,rects);self.phase='travel_place'
     continue
    return a
   if self.phase in ['travel_pick','travel_place']:
    if self.path and np.linalg.norm(self.path[0]-base)<.005:self.path.pop(0)
    if self.path:
     delta=self.path[0]-base;a[:2]=delta*min(1.,.2/max(np.max(abs(delta)),1e-8))
    a[2]=np.clip(self.wrap(self.dyaw-yaw),-.2,.2)
    if not self.path and abs(self.wrap(self.dyaw-yaw))<.003:self.phase='lower_pick' if self.phase=='travel_pick' else 'lower_place';continue
    if self.stuck>5:
     # Move the base around a corner before retrying the remaining route.
     a[:2]=[0,.2 if base[1]>=tab[1] else -.2];self.stuck=0
    return a
   if self.phase in ['lower_pick','lower_place']:
    zz=self.pick[2] if self.phase=='lower_pick' else self.place[2]
    if self.phase=='lower_place':
     tf=np.array([s.get(r,'grasp_tf_'+v) for v in 'xyz']);zz+=tf[2]
    current_z=fk(q)[2,3]+.3948
    increment=.09 if self.count>12 and self.h<.06 else .06
    des=self.q_for(max(zz,current_z-increment))
    speed=.16 if self.count>12 and self.h<.06 else .12
    delta=des-q;a[3:10]=delta*min(1.,speed/max(np.max(abs(delta)),1e-8));a[10]=1 if self.phase=='lower_pick' else 0
    if np.max(abs(des-q))<.003 and abs(current_z-zz)<.003:self.phase='grasp' if self.phase=='lower_pick' else 'release';continue
    return a
   if self.phase=='grasp':
    a[10]=-1
    if s.get(r,'grasp_active') or s.get(o,'grasp_active'):self.phase='lift';continue
    return a
   if self.phase=='release':
    a[10]=1
    if not s.get(r,'grasp_active') and not s.get(o,'grasp_active'):
     self.finished.add(self.target);self.phase='select'
    return a
  return a
