import numpy as np
from env_client import make_env

def vals(s):
 out={}
 for name in s.get_object_names():
  o=s.get_object_from_name(name)
  feats=['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(i) for i in range(1,8)]+['pos_gripper'] if name=='robot' else ['x','y','z']
  out[name]=[round(float(s.get(o,f)),5) for f in feats]
 return out

def main():
 e=make_env();s,info=e.reset(seed=0)
 print('info',repr(info));print('initial',vals(s),flush=True)
 for i in range(11):
  s,_=e.reset(seed=0)
  a=np.zeros(11,np.float32);a[i]=1 if i==10 else .1
  for j in range(3):s,r,t,tr,inf=e.step(a)
  print('axis',i,'state',vals(s),'reward',r,'info',repr(inf),flush=True)
 e.close()
if __name__=='__main__':main()
