import math
import time
import numpy as np
from planner import Planner, rectangle, wrap

class GeneratedApproach:
 def __init__(self,action_space,observation_space,primitives):
  self.types={t.name:t for t in observation_space.types}
  self.low=np.asarray(action_space.low);self.high=np.asarray(action_space.high)
 def reset(self,state,info):
  self.rng=np.random.default_rng(123)
  self.robot=state.get_objects(self.types['crv_robot'])[0]
  sh=state.get_objects(self.types['shelf'])[0]
  self.shelf=tuple(state.get(sh,f) for f in ['x1','y1','width1','height1'])
  self.finished=set();self.target=None;self.path=[];self.phase='choose';self.held=None
  self.lastq=None;self.stuck=0;self.failures={};self.steps=0
  self.slots={};self.initial=None
 def read(self,state):
  self.q=np.array([state.get(self.robot,f) for f in ['x','y','theta','arm_joint']])
  self.blocks={o.name:rectangle(*(state.get(o,f) for f in ['x','y','theta','width','height'])) for o in state.get_objects(self.types['target_block'])}
 def planner(self,exclude=None):
  p=Planner({k:b for k,b in self.blocks.items() if k!=exclude},self.shelf)
  p.held=self.held
  p.deadline=self.deadline
  return p
 def contained(self,b):
  ex=abs(math.cos(b[2]))*b[3]+abs(math.sin(b[2]))*b[4]
  ey=abs(math.sin(b[2]))*b[3]+abs(math.cos(b[2]))*b[4]
  sx,sy,sw,sh=self.shelf
  return b[0]-ex>=sx-.002 and b[0]+ex<=sx+sw+.002 and b[1]-ey>=sy-.002 and b[1]+ey<=sy+sh+.002
 def assign_slot(self,k,reserve=True):
  if k in self.slots:return self.slots[k]
  occupied=set(self.slots.values());candidates=[]
  blocked={self.slots[n][0] for n in self.initial if n not in self.finished}
  maxrow=int((self.top_y-self.shelf[1]-self.blocks[k][4]-.005)/self.row_pitch)
  for col in range(self.cols):
   row=0
   while (col,row) in occupied:row+=1
   if row<=maxrow and (col not in blocked or len(blocked)==self.cols):
    x=self.shelf[0]+(col+.5)*self.shelf[2]/self.cols
    candidates.append((abs(x-self.blocks[k][0])+.04*row,col,row))
  if candidates:
   _,col,row=min(candidates);slot=(col,row)
  else:
   slot=next((c,r) for r in range(len(self.blocks)+1) for c in range(self.cols) if (c,r) not in occupied)
  if reserve:self.slots[k]=slot
  return slot
 def act(self,d,vac):return np.clip(np.array([*d,vac]),self.low,self.high).astype(np.float32)
 def goto(self,goal,vac):
  d=goal-self.q;d[2]=wrap(d[2]);scale=max(1.,abs(d[0])/.05,abs(d[1])/.05,abs(d[2])/.19634954,abs(d[3])/.1)
  return self.act(d/scale,vac)
 def get_action(self,state):
  self.read(state);self.steps+=1
  self.deadline=time.perf_counter()+.8
  self.finished={k for k in self.finished if k in self.blocks and self.contained(self.blocks[k])}
  if self.initial is None:
   self.initial={k for k,b in self.blocks.items() if b[1]>self.shelf[1]}
   maxw=max((b[3]*2 for b in self.blocks.values()),default=.28)
   maxh=max((b[4]*2 for b in self.blocks.values()),default=.04)
   self.cols=max(1,int(self.shelf[2]/(maxw+.015)))
   rows=max(1,math.ceil(len(self.blocks)/self.cols))
   self.row_pitch=min(.076,(self.shelf[3]-maxh-.025)/max(1,rows-1))
   self.top_y=self.shelf[1]+self.shelf[3]-maxh/2-.011
   keys=sorted(self.initial,key=lambda k:self.blocks[k][0])
   for i,k in enumerate(keys):self.slots[k]=(i%self.cols,0)
  if self.lastq is not None and np.max(abs(self.q-self.lastq))<1e-6:self.stuck+=1
  else:self.stuck=0
  self.lastq=self.q.copy()
  for loop in range(8):
   if self.path:
    goal=self.path[0];d=goal-self.q;d[2]=wrap(d[2])
    if np.max(abs(d))<.0002:self.path.pop(0);continue
    if self.stuck>4:
     self.path=[];self.stuck=0
     if self.held is None:self.phase='choose';self.failures[self.target]=self.failures.get(self.target,0)+1
     else:self.phase='deliver'
     continue
    vac=1 if self.held is not None else 0
    if self.phase=='grasp' and len(self.path)==1 and np.all(np.abs(d)<=self.high[:4]+1e-7):vac=1
    return self.goto(goal,vac)
   if self.phase=='choose':
    if self.target not in self.finished and self.target not in self.initial:self.slots.pop(self.target,None)
    pending=[k for k in self.blocks if k not in self.finished]
    if not pending:return self.act([0,0,0,0],0)
    pending.sort(key=lambda k:(self.failures.get(k,0),k not in self.initial,math.hypot(self.blocks[k][0]-self.q[0],self.blocks[k][1]-self.q[1])))
    found=False
    for k in pending:
     if time.perf_counter()>self.deadline:break
     b=self.blocks[k];bx,by,bt,bw,bh=b
     col,row=self.assign_slot(k,reserve=False)
     destx=self.shelf[0]+(col+.5)*self.shelf[2]/self.cols
     lateral=.075 if destx<.25 else (-.075 if destx>4.75 else 0.)
     if self.failures.get(k,0)>0:
      offsets=[v for v in [lateral,.07,-.07] if .205<=destx+v<=4.795]
      lateral=offsets[self.failures[k]%len(offsets)] if offsets else lateral
     ts=[wrap(bt+math.pi/2),wrap(bt-math.pi/2)]
     ts.sort(key=lambda t:-math.sin(t))
     ts += [wrap(t+bias) for t in ts[:] for bias in [-.12,.12]]
     for t in ts:
      c,s=math.cos(t),math.sin(t)
      for arm in [.5,.7,.3,.2]:
       dist=bh*abs(math.sin(t-bt))+bw*abs(math.cos(t-bt))+.028
       goal=np.array([bx-(arm+dist)*c+lateral*s,by-(arm+dist)*s-lateral*c,t,arm])
       p=self.planner();path=p.plan(self.q,goal,self.rng,limit=550)
       if path is not None:
        self.slots[k]=(col,row);self.target=k;self.path=path;self.phase='grasp';self.grasp_start=self.blocks[k];self.grasp_tries=0;found=True;break
      if found:break
     if found:break
     self.failures[k]=self.failures.get(k,0)+1
    if not found:
     p=self.planner()
     candidates=[np.array([0,0,0,-.1]),np.array([0,0,.15,0]),np.array([0,0,-.15,0])]
     candidates += [np.array([dx,dy,0,0]) for dx,dy in [(0,-.05),(.05,0),(-.05,0),(0,.05),(.035,-.035),(-.035,-.035)]]
     if self.stuck>3:self.rng.shuffle(candidates)
     for delta in candidates:
      if abs(delta[3])>0 and self.q[3]<.201:continue
      if p.edge(self.q,self.q+delta):return self.act(delta,0)
     return self.act([0,-.015,0,-.02],0)
    continue
   if self.phase=='grasp':
    b=self.blocks[self.target];old=self.grasp_start
    if math.hypot(b[0]-old[0],b[1]-old[1])>.0005:
     x,y,t,d=self.q;c,s=math.cos(t),math.sin(t);gx,gy=x+d*c,y+d*s
     dx,dy=b[0]-gx,b[1]-gy
     self.held=(dx*c+dy*s,-dx*s+dy*c,wrap(b[2]-t),b[3],b[4]);self.phase='deliver';continue
    self.grasp_tries+=1
    if self.grasp_tries>18:
     self.failures[self.target]=self.failures.get(self.target,0)+1;self.phase='choose';return self.act([0,0,0,-.04],0)
    return self.act([0,0,0,.002,1][:4],1)
   if self.phase=='deliver':
    k=self.target
    col,row=self.assign_slot(k)
    gx=self.shelf[0]+(col+.5)*self.shelf[2]/self.cols;gy=self.top_y-row*self.row_pitch
    hx,hy,ht,hw,hh=self.held
    target_t=wrap(-ht)
    if math.sin(target_t)<0:target_t=wrap(target_t+math.pi)
    c,s=math.cos(target_t),math.sin(target_t)
    found=False
    for arm in [.6,.75,.45,.3,.2]:
     goal=np.array([gx-(arm+hx)*c+hy*s,gy-(arm+hx)*s-hy*c,target_t,arm])
     p=self.planner(k)
     entry=goal.copy();entry[1]-=max(0.,gy-(self.shelf[1]-.145))
     if k not in self.initial and p.edge(entry,goal):
      path=p.plan(self.q,entry,self.rng,limit=1500)
      if path is not None:path.append(goal)
     else:path=p.plan(self.q,goal,self.rng,limit=1500)
     if path is not None:
      self.path=path;self.phase='release';found=True;break
    if not found:
     # Withdraw into open workspace before retrying transport.
     p=self.planner(k)
     for _ in range(30):
      goal=self.q+np.array([self.rng.uniform(-.3,.3),self.rng.uniform(-.3,.05),self.rng.uniform(-.5,.5),0])
      if p.edge(self.q,goal):self.path=[goal];break
     if not self.path:return self.act([0,-.01,0,0],1)
    continue
   if self.phase=='release':
    if self.contained(self.blocks[self.target]):self.finished.add(self.target)
    else:self.failures[self.target]=self.failures.get(self.target,0)+1
    self.held=None;self.phase='retreat';return self.act([0,0,0,0],0)
   if self.phase=='retreat':
    self.phase='choose';continue
  return self.act([0,0,0,0],1 if self.held else 0)
