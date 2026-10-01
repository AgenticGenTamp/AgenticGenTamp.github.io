import math
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
from planner import wrap,overlap

def diagnose(p,q):
 x,y,t,d=q;c,s=math.cos(t),math.sin(t)
 reasons=[]
 if x<p.radius+.001 or x>5-p.radius-.001 or y<p.radius+.001 or y>3-p.radius-.001 or d<.1999 or d>.8001:reasons.append(('basebounds',q.tolist()))
 gx,gy=x+d*c,y+d*s
 parts=[(x+d*c/2,y+d*s/2,t,d/2,.008),(gx,gy,t,p.gw/2,p.gh/2)]
 hx,hy,ht,hw,hh=p.held;parts.append((gx+hx*c-hy*s,gy+hx*s+hy*c,t+ht,hw,hh))
 for i,a in enumerate(parts):
  xx,yy,tt,ww,hh=a;cc,ss=abs(math.cos(tt)),abs(math.sin(tt));ex,ey=cc*ww+ss*hh,ss*ww+cc*hh
  if xx-ex<.001 or xx+ex>4.999 or yy-ey<.001 or yy+ey>2.999:reasons.append(('partbounds',i,a))
 for j,b in enumerate(p.obs):
  bx,by,bt,bw,bh=b;bc,bs=math.cos(bt),math.sin(bt);dx,dy=x-bx,y-by
  ux=max(abs(dx*bc+dy*bs)-bw,0);uy=max(abs(-dx*bs+dy*bc)-bh,0)
  if ux*ux+uy*uy<(p.radius+.002)**2:reasons.append(('basecollision',j,b))
  for i,a in enumerate(parts):
   if overlap(a,b,.001):reasons.append(('partcollision',i,j,b))
 return reasons

for seed in [135,219]:
 env=make_env();state,info=env.reset(seed=seed);a=GeneratedApproach(env.action_space,env.observation_space,{});a.reset(state,info)
 for step in range(200):
  act=a.get_action(state)
  if a.held is not None:
   print('SEED',seed,'STEP',step,'PHASE',a.phase,'Q',a.q,'HELD',a.held,'SHELF',a.shelf,'SLOTS',a.slots,'BLOCKS',a.blocks,flush=True)
   col,row=a.slots[a.target];gx=a.shelf[0]+(col+.5)*a.shelf[2]/a.cols;gy=a.top_y-row*a.row_pitch
   hx,hy,ht,hw,hh=a.held;t=wrap(-ht)
   if math.sin(t)<0:t=wrap(t+math.pi)
   c,s=math.cos(t),math.sin(t);p=a.planner(a.target)
   for arm in [.6,.75,.45]:
    q=np.array([gx-(arm+hx)*c+hy*s,gy-(arm+hx)*s-hy*c,t,arm])
    print('GOAL',q,'VALID',p.valid(q),'REASONS',diagnose(p,q),flush=True)
   break
  state,_,term,trunc,_=env.step(act)
 env.close()
