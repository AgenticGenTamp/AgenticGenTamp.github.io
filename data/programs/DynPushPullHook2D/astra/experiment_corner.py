from env_client import make_env
from baseline_grasp import GeneratedApproach
import numpy as np,math,argparse
p=argparse.ArgumentParser();p.add_argument('--angle',type=float,default=1.8);p.add_argument('--dx',type=float,default=.01);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
e=make_env();s,info=e.reset(seed=a.seed);g=GeneratedApproach(e.action_space,e.observation_space,{});g.reset(s,info)
r=s.get_objects(g.rt)[0];h=s.get_objects(g.ht)[0];t=s.get_objects(g.tt)[0];phase=0
for i in range(350):
 rx,ry,rt=[s.get(r,f) for f in ['x','y','theta']];hx,hy,ht=[s.get(h,f) for f in ['x','y','theta']];tx,ty,tt=[s.get(t,f) for f in ['x','y','theta']]
 hw=(abs(math.cos(tt))*s.get(t,'width')+abs(math.sin(tt))*s.get(t,'height'))/2
 if not s.get(h,'held'):action=g.get_action(s)
 else:
  theta=rt+(a.angle-ht+math.pi)%(2*math.pi)-math.pi;d=theta-rt
  ox=math.cos(d)*(hx-rx)-math.sin(d)*(hy-ry);oy=math.sin(d)*(hx-rx)+math.cos(d)*(hy-ry)
  if phase==0:
   gx=rx;gy=1.47
   if abs(ht-a.angle)<.02 and abs(ry-1.47)<.02:phase=1
  elif phase==1:
   gx=tx+hw+.12-ox;gy=1.47
   if abs(rx-gx)<.02:phase=2
  else:
   gx=rx-a.dx;gy=max(.25,ry-.02)
  action=g.move(s,r,np.clip(gx,.25,3.25),gy,theta,gap=.12)
 s,rew,done,trunc,_=e.step(action)
 if i%20==0 or done:print(i,phase,'h',round(hx,2),round(hy,2),'t',round(tx,2),round(ty,2),'done',done,flush=True)
 if done:break
e.close()
