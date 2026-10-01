from env_client import make_env
from approach import GeneratedApproach
from types import SimpleNamespace
import numpy as np,time
E=make_env();base,_=E.reset(seed=0)
class Synthetic:
 def __init__(self,n):
  self.robot=base.get_object_from_name('robot');self.stick=base.get_object_from_name('stick')
  self.buttons=[SimpleNamespace(name='button'+str(i)) for i in range(n)]
  self.data={b.name:dict(x=float(x),y=float(y),color_g=0.) for b,(x,y) in zip(self.buttons,np.random.default_rng(3).uniform([.05,.05],[3.45,2.45],(n,2)))}
 def get_objects(self,t):
  return self.buttons if t.name=='circle' else [self.robot] if t.name=='crv_robot' else [self.stick] if t.name=='rectangle' else []
 def get(self,o,f):
  return self.data[o.name][f] if o.name in self.data else base.get(o,f)
for n in [100,1000,10000]:
 s=Synthetic(n);p=GeneratedApproach(E.action_space,E.observation_space,{})
 start=time.monotonic();p.reset(s,{});p.phase='held';a=p.get_action(s)
 print(n,'objects',round(time.monotonic()-start,4),'seconds','valid',E.action_space.contains(a) if hasattr(E.action_space,'contains') else bool(np.all(a>=E.action_space.low) and np.all(a<=E.action_space.high)))
E.close()
