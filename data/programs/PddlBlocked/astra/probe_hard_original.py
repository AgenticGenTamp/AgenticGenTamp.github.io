from env_client import make_env
from kinematics import solve,fk
import numpy as np, math
fs=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]
e=make_env();s,_=e.reset(seed=42)
def pos(name):
 o=s.get_object_from_name(name);return np.array([s.get(o,'pose_'+f) for f in 'xyz'])
g=pos('green0');b=pos('blocker');d=(g-b)/np.linalg.norm(g-b);yaw=math.atan2(d[1],d[0]);print('SCENE',g,b,yaw,flush=True)
r=s.get_object_from_name('robot')
def curr():return np.array([s.get(r,f) for f in fs])
def move(v,grip=1):
 global s
 for _ in range(80):
  p=curr();delta=v-p;delta[[2,7,9]]=(delta[[2,7,9]]+math.pi)%(2*math.pi)-math.pi
  if max(abs(delta))<1e-5:
   a=np.zeros(11);a[10]=grip;s,*_=e.step(a);return True
  a=np.zeros(11);a[:10]=delta*min(1,.2/max(abs(delta)));a[10]=grip;s,*_=e.step(a)
  if max(abs(curr()-p))<1e-6:return False
 return False
for bx in [3.73,3.65,3.55]:
 for by in np.arange(-.8,.91,.2):
  for roll in [0,math.pi]:
   cfg,err=solve(g-.035*d+np.array([0,0,.03]),yaw,[bx,by])
   if err>.005:continue
   cfg[9]=(cfg[9]+roll+math.pi)%(2*math.pi)-math.pi
   cp=cfg.copy();cp[:2]-=.32*d[:2];cr=cfg.copy();cr[:2]-=.15*d[:2];high=cp.copy();high[4]-=.3
   s,_=e.reset(seed=42);v=curr();v[0]=3.0
   stages=[v.copy()];v[1]=cp[1];stages.append(v.copy());v[2:]=high[2:];stages.append(v.copy());v[:2]=cp[:2];stages.append(v.copy());stages += [high,cp,cr]
   for idx,v in enumerate(stages):
    ok=move(v,-1 if idx==len(stages)-1 else 1)
    if not ok:break
   held=s.get(r,'grasp_active')
   print('TRY',bx,round(by,2),round(roll,2),'stage',idx,'ok',ok,'held',held,'tool',fk(curr())[0].round(3).tolist(),flush=True)
   if held:
    print('CONFIG',cfg.tolist(),flush=True)
    ok=move(cp,0) and move(high,0);v=high.copy();v[:2]+=[-.65,.65];ok=ok and move(v,1)
    ok=ok and move(high,1) and move(cp,1) and move(cr,1) and move(cfg,-1)
    print('GREEN',ok,s.get(r,'grasp_active'),'robot',curr().tolist(),flush=True)
    if ok and s.get(r,'grasp_active'):e.close();quit()
e.close()
