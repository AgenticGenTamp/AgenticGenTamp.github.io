import math
import time
import numpy as np
from geometry import Scene
from planner import Planner,wrap,difference,metric

class GeneratedApproach:
 def __init__(self,action_space,observation_space,primitives):
  self.action_space=action_space
  self.types={t.name:t for t in observation_space.types}
 def reset(self,state,info):
  self.rng=np.random.default_rng(123)
  self.deadline=time.monotonic()+48.
  self.phase='choose';self.path=[];self.prep_path=[];self.chosen=None;self.held=None
  self.old_positions=None;self.previous_q=None;self.stuck=0;self.steps=0;self.recoveries=0
  self.failures={};self.removed=set();self.planning_time=0.;self.failed_goals={}
 def read(self,s):
  self.robot=next(iter(s.get_objects(self.types['crv_robot'])))
  self.target=next(iter(s.get_objects(self.types['target_block'])))
  region=next(iter(s.get_objects(self.types['target_region'])))
  self.objects=sorted([o for o in s.get_objects(self.types['rectangle']) if o!=region],key=lambda o:o.name)
  self.rects=np.array([[s.get(o,f) for f in ['x','y','theta','width','height']] for o in self.objects])
  self.rects[:,:2]+=np.column_stack((np.cos(self.rects[:,2])*self.rects[:,3]/2-np.sin(self.rects[:,2])*self.rects[:,4]/2,np.sin(self.rects[:,2])*self.rects[:,3]/2+np.cos(self.rects[:,2])*self.rects[:,4]/2))
  self.region=np.array([s.get(region,f) for f in ['x','y','theta']])
  a=self.region[2];w=s.get(region,'width')/2;h=s.get(region,'height')/2
  self.region[:2]+=np.array([math.cos(a)*w-math.sin(a)*h,math.sin(a)*w+math.cos(a)*h])
  self.q=np.array([s.get(self.robot,f) for f in ['x','y','theta']])
  self.arm=s.get(self.robot,'arm_joint')
  self.scene=Scene(self.rects)
  self.planner=Planner(self.scene,self.rng)
 def plan(self,goals,ignore=None,held=None,budget=.35):
  if self.planning_time>36 or time.monotonic()>self.deadline:return None
  start=time.monotonic()
  p=self.planner.plan(self.q,goals,ignore,held,budget)
  self.planning_time+=time.monotonic()-start
  return p
 def grasp_goals(self,idx):
  ob=self.rects[idx];goals=[]
  for k in range(4):
   angle=wrap(ob[2]+k*math.pi/2)
   u=np.array([math.cos(angle),math.sin(angle)])
   side=ob[3 if k%2==0 else 4]/2
   for tangent in [0,-.025,.025]:
    pos=ob[:2]-u*(.207+side)+np.array([-u[1],u[0]])*tangent
    q=np.r_[pos,angle]
    # Ignore only gripper contact with desired object, base is far away.
    if self.scene.valid(np.r_[q,.2]) and all(metric(q,g)>.09 for g in self.failed_goals.get(self.objects[idx].name,[])):goals.append(q)
  return goals
 def choose(self):
  ti=self.objects.index(self.target)
  blockers=[]
  for i,rect in enumerate(self.rects):
   if i==ti:continue
   d=self.region[:2]-rect[:2];a=rect[2]
   local=np.abs([math.cos(a)*d[0]+math.sin(a)*d[1],-math.sin(a)*d[0]+math.cos(a)*d[1]])-rect[3:]/2
   if np.linalg.norm(np.maximum(local,0))<.105:blockers.append(i)
  order=blockers+[ti]+sorted([i for i in range(len(self.objects)) if i!=ti and i not in blockers],key=lambda i: (self.objects[i].name in self.removed,self.failures.get(self.objects[i].name,0),np.linalg.norm(self.rects[i,:2]-self.rects[ti,:2])+.2*np.linalg.norm(self.rects[i,:2]-self.q[:2])))
  for idx in order:
   if self.failures.get(self.objects[idx].name,0)>4:continue
   goals=self.grasp_goals(idx)
   path=self.plan(goals,budget=.55 if idx==ti else .18)
   if path:
    self.chosen=self.objects[idx].name;self.path=path;self.grasp_goal=path[-1].copy();self.phase='approach';self.held=None;self.creeps=0
    return True
  self.failures={};return False
 def carry_goals(self,idx):
  goals=[]
  is_target=self.objects[idx]==self.target
  if is_target:positions=[self.region[:2]]
  else:
   ti=self.objects.index(self.target);t=self.rects[ti,:2]
   d=self.rects[idx,:2]-t;d/=max(np.linalg.norm(d),.001)
   positions=[self.rects[idx,:2]+d*.5]
   positions += [np.array([x,y]) for x in [.35,.8,1.25,1.7,2.15] for y in [.35,.8,1.25,1.7,2.15] if np.linalg.norm(np.array([x,y])-t)>.65 and np.linalg.norm(np.array([x,y])-self.region[:2])>.35]
   positions.sort(key=lambda p:np.linalg.norm(p-self.rects[idx,:2]))
  for pos in positions:
   for a in [self.q[2]]+list(np.linspace(-math.pi,math.pi,16,endpoint=False)):
    u=np.array([math.cos(a),math.sin(a)]);v=np.array([-u[1],u[0]])
    q=np.r_[pos-u*self.held[0]-v*self.held[1],a]
    if self.scene.valid(np.r_[q,.2],ignore=idx,held=self.held):goals.append(q)
   if not is_target and len(goals)>24:break
  return goals
 def get_action(self,state):
  self.steps+=1;self.read(state)
  if time.monotonic()>self.deadline+5:
   return np.zeros(5,dtype=np.float32)
  q=self.q
  if self.previous_q is not None and np.max(abs(difference(self.previous_q,q)))<1e-5:self.stuck+=1
  else:self.stuck=0
  if self.old_positions is not None and self.phase in ['grasp','probe','approach']:
   moved=np.linalg.norm(self.rects[:,:2]-self.old_positions,axis=1)
   if np.max(moved)>1e-5:
    idx=int(np.argmax(moved));self.chosen=self.objects[idx].name
    rel=self.rects[idx,:2]-q[:2];c=math.cos(q[2]);s=math.sin(q[2])
    self.held=np.array([c*rel[0]+s*rel[1],-s*rel[0]+c*rel[1],wrap(self.rects[idx,2]-q[2]),*self.rects[idx,3:]])
    self.phase='carry';self.path=[];self.stuck=0
  self.old_positions=self.rects[:,:2].copy();self.previous_q=q.copy()
  vac=1 if self.phase in ['grasp','probe','carry'] else 0
  if self.stuck>=3 and self.phase=='carry':
   u=np.array([math.cos(q[2]),math.sin(q[2])]);v=np.array([-u[1],u[0]])
   probes=[np.r_[-.025*u,0],np.array([0,0,.025]),np.r_[.025*v,0],np.r_[-.025*v,0],np.array([0,0,-.025]),np.r_[.012*u,0],np.array([.02,0,0]),np.array([-.02,0,0]),np.array([0,.02,0]),np.array([0,-.02,0])]
   d=probes[self.recoveries%len(probes)];self.recoveries+=1;self.path=[];self.stuck=0
   return np.array([*d,0,1],dtype=np.float32)
  if self.stuck>5 and self.phase in ['approach','carry']:
   if self.phase=='approach':
    self.failures[self.chosen]=self.failures.get(self.chosen,0)+1;self.failed_goals.setdefault(self.chosen,[]).append(self.grasp_goal);self.phase='choose';vac=0
   self.path=[];self.stuck=0
  for _ in range(5):
   if self.phase=='choose':
    if self.arm<.1999:
     if self.scene.valid(np.r_[q,.2]) and self.stuck<3:
      return np.array([0,0,0,.2-self.arm,0],dtype=np.float32)
     self.planner.arm=self.arm
     while self.prep_path and metric(q,self.prep_path[0])<.0001:self.prep_path.pop(0)
     if not self.prep_path or self.stuck>5:
      goals=[]
      for radius in [0,.15,.3,.5]:
       for a in np.linspace(-math.pi,math.pi,16,endpoint=False):
        p=q[:2]+radius*np.array([math.cos(a),math.sin(a)])
        dest=np.r_[p,wrap(q[2]+a)]
        if self.scene.valid(np.r_[dest,.2]):goals.append(dest)
      self.prep_path=self.plan(goals,budget=.3) or []
      self.stuck=0
     if self.prep_path:
      d=difference(q,self.prep_path[0]);scale=max(np.max(abs(d[:2]))/.05,abs(d[2])/.19634954,1.)
      return np.array([*(d/scale),0,0],dtype=np.float32)
     return np.array([*self.rng.uniform(-.025,.025,2),.15,0,0],dtype=np.float32)
    if not self.choose():
     return np.array([*self.rng.uniform(-.04,.04,2),self.rng.uniform(-.19,.19),0,0],dtype=np.float32)
    vac=0
   if self.phase in ['approach','carry']:
    idx=next(i for i,o in enumerate(self.objects) if o.name==self.chosen)
    if self.phase=='carry' and not self.path:
     goals=self.carry_goals(idx)
     self.path=self.plan(goals,idx,self.held,budget=.7) or []
     if not self.path:
      self.failures[self.chosen]=self.failures.get(self.chosen,0)+1
      self.failed_goals.setdefault(self.chosen,[]).append(self.grasp_goal)
      self.phase='choose';self.held=None
      return np.array([0,0,0,0,0],dtype=np.float32)
    while self.path and metric(q,self.path[0])<.0001:self.path.pop(0)
    if self.path:
     d=difference(q,self.path[0]);scale=max(np.max(abs(d[:2]))/.05,abs(d[2])/.19634954,1.)
     return np.array([*(d/scale),0,vac],dtype=np.float32)
    if self.phase=='approach':self.phase='grasp';self.creeps=0;vac=1
    else:
     self.removed.add(self.chosen);self.phase='choose';self.held=None;self.failures={};self.failed_goals={}
     return np.array([0,0,0,0,0],dtype=np.float32)
   if self.phase=='grasp':
    self.creeps+=1
    if self.creeps>50:
     self.failures[self.chosen]=self.failures.get(self.chosen,0)+1;self.phase='choose'
     return np.array([-.035*math.cos(q[2]),-.035*math.sin(q[2]),0,0,0],dtype=np.float32)
    if self.stuck>=5:
     probes=[[0,0,.012,0,1],[0,0,-.012,0,1],[-.005*math.cos(q[2]),-.005*math.sin(q[2]),0,0,1],[.005,0,0,0,1],[0,.005,0,0,1],[-.005,0,0,0,1],[0,-.005,0,0,1]]
     return np.array(probes[(self.stuck-5)%len(probes)],dtype=np.float32)
    step=.004 if self.stuck<2 else .0002
    return np.array([step*math.cos(q[2]),step*math.sin(q[2]),0,0,1],dtype=np.float32)
  return np.zeros(5,dtype=np.float32)
