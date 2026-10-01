import math
import time
import numpy as np

PI=math.pi
def wrap(a):return (a+PI)%(2*PI)-PI

def rectangle(x,y,t,w,h):
 c,s=math.cos(t),math.sin(t)
 return (x+c*w/2-s*h/2,y+s*w/2+c*h/2,t,w/2,h/2)

def overlap(a,b,margin=0.):
 ax,ay,at,aw,ah=a;bx,by,bt,bw,bh=b
 ac,ass=math.cos(at),math.sin(at);bc,bs=math.cos(bt),math.sin(bt)
 dx,dy=bx-ax,by-ay
 cc=abs(ac*bc+ass*bs);ss=abs(ac*bs-ass*bc)
 return (abs(dx*ac+dy*ass)<aw+bw*cc+bh*ss+margin and abs(-dx*ass+dy*ac)<ah+bw*ss+bh*cc+margin and abs(dx*bc+dy*bs)<bw+aw*cc+ah*ss+margin and abs(-dx*bs+dy*bc)<bh+aw*ss+ah*cc+margin)

class Planner:
 def __init__(self,blocks,shelf,radius=.2,gh=.14,gw=.02):
  self.blocks=blocks;self.shelf=shelf;self.radius=radius;self.gh=gh;self.gw=gw
  self.obs=list(blocks.values())
  x,y,w,h=shelf
  if x>0:self.obs.append((x/2,y+h/2,0,x/2,h/2))
  if x+w<5:self.obs.append(((x+w+5)/2,y+h/2,0,(5-x-w)/2,h/2))
  self.held=None;self.deadline=float("inf")
 def valid(self,q):
  x,y,t,d=q;c,s=math.cos(t),math.sin(t)
  if x<self.radius+.001 or x>5-self.radius-.001 or y<self.radius+.001 or y>3-self.radius-.001 or d<.1999 or d>.8001:return False
  gx,gy=x+d*c,y+d*s
  # Thin radial arm and perpendicular rectangular gripper.
  parts=[(x+d*c/2,y+d*s/2,t,d/2,.008),(gx,gy,t,self.gw/2,self.gh/2)]
  if self.held is not None:
   hx,hy,ht,hw,hh=self.held
   parts.append((gx+hx*c-hy*s,gy+hx*s+hy*c,t+ht,hw,hh))
  for a in parts:
   xx,yy,tt,ww,hh=a;cc,ss=abs(math.cos(tt)),abs(math.sin(tt))
   ex,ey=cc*ww+ss*hh,ss*ww+cc*hh
   if xx-ex<.001 or xx+ex>4.999 or yy-ey<.001 or yy+ey>2.999:return False
  for b in self.obs:
   bx,by,bt,bw,bh=b;bc,bs=math.cos(bt),math.sin(bt)
   dx,dy=x-bx,y-by
   ux=max(abs(dx*bc+dy*bs)-bw,0);uy=max(abs(-dx*bs+dy*bc)-bh,0)
   if ux*ux+uy*uy<(self.radius+.002)**2:return False
   for a in parts:
    if overlap(a,b,.001):return False
  return True
 def delta(self,a,b):
  d=np.asarray(b)-a;d[2]=wrap(d[2]);return d
 def edge(self,a,b):
  d=self.delta(a,b);n=max(1,int(math.ceil(max(abs(d[0])/.012,abs(d[1])/.012,abs(d[2])/.035,abs(d[3])/.025))))
  return all(self.valid(a+d*(j/n)) for j in range(1,n+1))
 def plan(self,start,goal,rng,limit=1800):
  start=np.array(start,float);goal=np.array(goal,float)
  if not self.valid(goal):return None
  if self.edge(start,goal):return [goal]
  # Bidirectional rapidly-exploring trees in robot configuration space.
  trees=[[start],[goal]];pars=[[-1],[-1]]
  scale=np.array([1,1,.48,1.])
  for it in range(limit):
   if it%16==0 and time.perf_counter()>self.deadline:return None
   side=it%2;other=1-side
   if rng.random()<.18:target=trees[other][-1]
   else:target=np.array([rng.uniform(.21,4.79),rng.uniform(.21,2.65),rng.uniform(-PI,PI),rng.uniform(.2,.8)])
   arr=np.array(trees[side]);dd=arr-target;dd[:,2]=(dd[:,2]+PI)%(2*PI)-PI
   idx=int(np.argmin(np.sum((dd*scale)**2,axis=1)));a=arr[idx];d=self.delta(a,target)
   norm=np.linalg.norm(d*scale)
   b=a+d*min(1.,.24/max(norm,1e-9));b[2]=wrap(b[2])
   if not self.edge(a,b):continue
   trees[side].append(b);pars[side].append(idx);bi=len(trees[side])-1
   arr2=np.array(trees[other]);dd=arr2-b;dd[:,2]=(dd[:,2]+PI)%(2*PI)-PI
   idx2=int(np.argmin(np.sum((dd*scale)**2,axis=1)))
   if not self.edge(arr2[idx2],b):continue
   def trace(k,ii):
    out=[]
    while ii!=-1:out.append(trees[k][ii]);ii=pars[k][ii]
    return out[::-1]
   aa=trace(side,bi);bb=trace(other,idx2)
   path=aa+bb[::-1] if side==0 else bb+aa[::-1]
   out=[];i=0
   while i<len(path)-1:
    j=len(path)-1
    while j>i+1 and not self.edge(path[i],path[j]):j-=1
    out.append(path[j]);i=j
   return out
  return None
