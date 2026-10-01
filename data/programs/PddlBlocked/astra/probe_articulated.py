from env_client import make_env
from kinematics import solve,fk
import numpy as np,math,sys
fs=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]
for seed in [int(x) for x in sys.argv[1:]] or [9,13]:
 e=make_env();s,_=e.reset(seed=seed);r=s.get_object_from_name('robot')
 def pos(name):
  o=s.get_object_from_name(name);return np.array([s.get(o,'pose_'+f) for f in 'xyz'])
 g=pos('green0');b=pos('blocker');d=(g-b)/np.linalg.norm(g-b);yaw=math.atan2(d[1],d[0]);print('SCENE',seed,g,b,yaw,flush=True)
 def curr():return np.array([s.get(r,f) for f in fs])
 def move(v,grip=0):
  global s
  for _ in range(100):
   p=curr();delta=v-p;delta[[2,7,9]]=(delta[[2,7,9]]+math.pi)%(2*math.pi)-math.pi
   a=np.zeros(11);a[:10]=delta*min(1,.2/max(1e-9,max(abs(delta))));a[10]=grip if max(abs(delta))<.20001 else 0
   s,*_=e.step(a)
   if max(abs(delta))<1e-5:return True
   if max(abs(curr()-p))<1e-6:return False
  return False
 done=False
 for xy in [[g[0]+dx,y] for y in [1.08,1.02,1.15] for dx in [.35,.15,0,-.2]]+[[x,g[1]+dy] for x in [3.7,3.6] for dy in [-.2,0,.2]]:
  if done:break
  for roll in [0,math.pi]:
   cr,er=solve(b-.035*d+np.array([0,0,.03]),yaw,xy)
   cg,eg=solve(g-.035*d+np.array([0,0,.03]),yaw,xy,cr)
   if max(er,eg)>.005:continue
   cr[9]=(cr[9]+roll+math.pi)%(2*math.pi)-math.pi;cg[9]=(cg[9]+roll+math.pi)%(2*math.pi)-math.pi
   high=cr.copy();high[4]-=.35;s,_=e.reset(seed=seed)
   v=curr();v[0]=3.0;stages=[v.copy()];v[1]=1.5 if xy[1]>.95 else xy[1];stages.append(v.copy());v[2:]=high[2:];stages.append(v.copy());v[0]=xy[0];stages.append(v.copy());v[:2]=xy;stages+=[v.copy(),high,cr]
   for idx,v in enumerate(stages):
    ok=move(v,-1 if idx==len(stages)-1 else 1)
    if not ok:break
   h=s.get(r,'grasp_active');print('TRY',seed,xy,roll,'stage',idx,ok,'held',h,flush=True)
   if not h:continue
   ok=move(high);v=high.copy();v[:2]+=[-.65,.65];ok=ok and move(v,1)
   print('DROP',ok,s.get(r,'grasp_active'),flush=True)
   if not ok:continue
   for idx,v in enumerate([high,cr,cg]):
    ok=move(v,-1 if idx==2 else 1)
    if not ok:break
   h=s.get(r,'grasp_active');print('GREEN',idx,ok,h,flush=True)
   if h:
    print('SUCCESS',seed,'cr',cr.tolist(),'cg',cg.tolist(),'extract',move(high),flush=True);done=True;break
 e.close()
