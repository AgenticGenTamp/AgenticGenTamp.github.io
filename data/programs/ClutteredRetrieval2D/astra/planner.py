import math
import time
import numpy as np

def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi

def difference(a,b):
 d=np.array(b)-a;d[2]=wrap(d[2]);return d

def metric(a,b):
 d=difference(a,b);return float(np.linalg.norm(d*np.array([1,1,.25])))

class Planner:
 def __init__(self,scene,rng):self.scene=scene;self.rng=rng;self.arm=.2
 def edge(self,a,b,ignore=None,held=None):
  d=difference(a,b)
  n=max(1,int(math.ceil(max(np.max(abs(d[:2]))/.025,abs(d[2])/.09))))
  qs=a+np.arange(1,n+1)[:,None]/n*d
  return bool(np.all(self.scene.valid_many(np.column_stack((qs,np.full(n,self.arm))),ignore=ignore,held=held)))
 def plan(self,start,goals,ignore=None,held=None,budget=1.0):
  if not goals:return None
  valid=lambda q:self.scene.valid(np.r_[q,self.arm],ignore=ignore,held=held)
  goals=[np.array(g) for g in goals if valid(g)]
  if not goals:return None
  goals.sort(key=lambda g:metric(start,g))
  for g in goals:
   if self.edge(start,g,ignore,held):return [g]
  end=time.monotonic()+budget
  # Bidirectional trees, with all acceptable grasp poses as roots.
  trees=[([np.array(start)],[-1]),(goals.copy(),[-1]*len(goals))]
  def near(nodes,q):
   arr=np.asarray(nodes);d=arr-q;d[:,2]=(d[:,2]+math.pi)%(2*math.pi)-math.pi
   return int(np.argmin(np.sum((d*np.array([1,1,.25]))**2,axis=1)))
  def extend(tree,q):
   nodes,parents=tree;i=near(nodes,q);d=difference(nodes[i],q)
   scale=max(np.max(abs(d[:2]))/.12,abs(d[2])/.45,1.)
   dest=nodes[i]+d/scale;dest[2]=wrap(dest[2])
   if not self.edge(nodes[i],dest,ignore,held):return None,False
   nodes.append(dest);parents.append(i);return len(nodes)-1,scale==1.
  def chain(tree,i):
   out=[]
   while i>=0:out.append(tree[0][i]);i=tree[1][i]
   return out
  turn=0
  while time.monotonic()<end:
   a=turn%2;b=1-a;turn+=1
   if self.rng.random()<.25:q=trees[b][0][int(self.rng.integers(len(trees[b][0])))]
   else:q=np.array([self.rng.uniform(.1,2.4),self.rng.uniform(.1,2.4),self.rng.uniform(-math.pi,math.pi)])
   ia,_=extend(trees[a],q)
   if ia is None:continue
   for _ in range(60):
    ib,reached=extend(trees[b],trees[a][0][ia])
    if ib is None:break
    if reached:
     ca=chain(trees[a],ia);cb=chain(trees[b],ib)
     path=(ca[::-1]+cb) if a==0 else (cb[::-1]+ca)
     # Greedy farthest visible shortcut.
     out=[];i=0
     while i<len(path)-1:
      j=len(path)-1
      while j>i+1 and not self.edge(path[i],path[j],ignore,held):j-=1
      out.append(path[j]);i=j
     return out
  return None
