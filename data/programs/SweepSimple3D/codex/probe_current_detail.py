import numpy as np
from env_client import make_env
from approach import GeneratedApproach

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def vec(s,n,fs): return np.array([v(s,n,f) for f in fs])

e=make_env();s,info=e.reset(seed=0,options={'object_count':1})
p=GeneratedApproach(e.action_space,e.observation_space,{})
p.reset(s,info); w0=vec(s,'wiper_0','xyz')
for k in range(100):
 a=p.get_action(s);s,r,d,t,_=e.step(a)
 if k%5==4 or np.linalg.norm(vec(s,'wiper_0','xyz')-w0)>.01:
  print(k+1,'base',vec(s,'robot',['pos_base_x','pos_base_y','pos_base_rot']).round(3),
   'q',vec(s,'robot',[f'pos_arm_joint{i}' for i in range(1,8)]).round(3),
   'g',round(v(s,'robot','pos_gripper'),3),'w',vec(s,'wiper_0','xyz').round(3),
   'quat',vec(s,'wiper_0',['qw','qx','qy','qz']).round(3),'r',r,flush=True)
e.close()
