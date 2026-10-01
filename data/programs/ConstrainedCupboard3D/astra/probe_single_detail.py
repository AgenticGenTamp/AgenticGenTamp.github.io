from env_client import make_env
from approach import GeneratedApproach
from kin import fk
from scipy.spatial.transform import Rotation
import numpy as np,sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 5
e=make_env();s,info=e.reset(seed=seed,options={'object_count':1});a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,info);last=None
for step in range(450):
 s,r,te,tr,_=e.step(a.get_action(s));status=(a.stage,len(a.queue));obj=a.objects[0]
 if status!=last or step%10==0 or te:
  p,R=fk(a.joints(s));b=np.array([s.get(a.robot,f) for f in ('pos_base_x','pos_base_y','pos_base_rot')]);op=a.xyz(s,obj);oq=[s.get(obj,f) for f in ('qx','qy','qz','qw')]
  print(step,status,'object',np.round(op,4).tolist(),'quat',np.round(oq,3).tolist(),'tool',np.round(p+[b[0]+.12,b[1],.4],4).tolist(),'done',te,flush=True);last=status
 if te or tr or a.stage==3:break
e.close()
