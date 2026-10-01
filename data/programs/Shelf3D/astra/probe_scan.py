from env_client import make_env
from approach import GeneratedApproach
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def run(grip):
 E=make_env();s,info=E.reset(seed=42);p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,info)
 o=p.objects[0];initial=np.array([s.get(o,f) for f in ['x','y','z']])
 for i in range(100):s,*_=E.step(p.get_action(s))
 count=0;best=0
 for k,dx in enumerate(np.arange(-.85,.9,.1)):
  for dy in (np.arange(-.85,.9,.1) if k%2==0 else np.arange(.85,-.9,-.1)):
   for repeat in range(4):
    a=p.get_action(s);a[-1]=grip
    a[:3]=np.clip(np.r_[initial[:2]+[dx,dy],0]-np.array([s.get(p.robot,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']]),-.1,.1)
    s,r,t,tr,info=E.step(a);count+=1
    xyz=np.array([s.get(o,f) for f in ['x','y','z']]);change=np.linalg.norm(xyz-initial)
    if change>max(best+.005,.005):
     best=change;print('GRIP',grip,'step',count,'baseoffset',dx,dy,'cube',xyz,'r',r,flush=True)
    if xyz[2]>.07:print('LIFT',grip,dx,dy,flush=True);E.close();return
 print('DONE',grip,'best',best,flush=True);E.close()
with ThreadPoolExecutor(2) as ex:list(ex.map(run,[0,1]))
