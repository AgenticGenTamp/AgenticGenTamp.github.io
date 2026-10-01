from env_client import make_env
from approach import GeneratedApproach
from kin import fk,planar_ik
import numpy as np
from concurrent.futures import ThreadPoolExecutor
class Columns(GeneratedApproach):
 def plan_place(self,s):
  fixture=self.fixtures[self.column]
  obj=self.objects[self.index]
  fp,fr=fk(self.joints(s))
  robotbase=np.array([s.get(self.robot,'pos_base_x'),s.get(self.robot,'pos_base_y'),.4])
  held=fr.T@(self.xyz(s,obj)-robotbase-np.array([.12,0,0])-fp)
  target=self.xyz(s,fixture)+[0,0,self.z]
  pitch=np.pi/2;yaw=np.pi/2
  _,rotation=fk(planar_ik(.65,-.1,yaw,pitch)[0])
  tool=target-rotation@held
  for pos in [tool+[-.5,0,0],tool]:
   q,err=planar_ik(.65,pos[2]-.4,yaw,pitch)
   base=np.array([pos[0]-.77,pos[1]-.00135,0.])
   self.queue.append((q,base,1,5))
  self.queue.append((q,base.copy(),0,8))
  self.queue.append((q,base+[-.4,0,0],0,5))
  self.stage=2
 def get_action(self,s):
  if self.stage==2 and self.waysteps>60:self.waysteps=181
  return super().get_action(s)
def trial(params):
 column,z=params
 e=make_env();s,info=e.reset(seed=0,options={'object_count':1})
 a=Columns(e.action_space,e.observation_space,{});a.column=column;a.z=z;a.reset(s,info)
 last=None;best=-1
 for t in range(500):
  act=a.get_action(s);s,r,te,tr,info=e.step(act);best=max(best,r)
  key=a.stage,len(a.queue)
  if key!=last or r>0 or te or tr:
   o=a.objects[0]; p=a.xyz(s,o);q=[s.get(o,k) for k in ['qw','qx','qy','qz']]
   print('COL',column,z,'t',t,'key',key,'p',np.round(p,4),'q',np.round(q,3),'r',r,'te',te,flush=True);last=key
  if te:
   print('FRAME',e.render_state(state=s,label='vertical_success_'+str(column)+'_'+str(z)),flush=True)
  if te or tr or a.stage==3:break
 print('END',column,z,'best',best,'info',info,flush=True);e.close()
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(trial,[(1,.2),(1,.25),(0,.45),(2,.16)]))
