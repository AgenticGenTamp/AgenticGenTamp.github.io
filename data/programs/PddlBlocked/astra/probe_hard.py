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
for bx in [3.65]:
 for by in [0]:
  for roll in [0]:
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
    for mode in ['q2-.02','q2-.05','q2-.1','q2-.2','up.005','up.02','up.05','up.1','up.2','back.01','back.05','side.05']:
     s,_=e.reset(seed=42)
     for idx,v in enumerate(stages):move(v,-1 if idx==len(stages)-1 else 1)
     p=curr();v=p.copy();tool,_=fk(p)
     if mode.startswith('q2'):v[4]+=float(mode[2:])
     elif mode.startswith('up'):v,err=solve(tool+np.array([0,0,float(mode[2:])]),yaw,p[:2],p)
     elif mode.startswith('back'):v[:2]-=float(mode[4:])*d[:2]
     else:v[0]-=.05
     ok=move(v,0)
     print('EXTRACT',mode,ok,'delta',(fk(curr())[0]-tool).round(5).tolist(),flush=True)
     if ok:
      v=curr();v[:2]+=[-.6,-.3];ok=move(v,1);print('DROP',mode,ok,s.get(r,'grasp_active'),flush=True)
      if ok and not s.get(r,'grasp_active'):
       for j,v in enumerate([high,cp,cr,cfg]):
        ok=move(v,-1 if j==3 else 1)
        if not ok:break
       print('GREEN',mode,ok,j,s.get(r,'grasp_active'),'p',curr().tolist(),flush=True)
       if ok and s.get(r,'grasp_active'):e.close();quit()
e.close()
