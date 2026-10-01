import numpy as np
from env_client import make_env

Q=np.array([3.87077975,1.62671101,-2.70210910,-.70062631,3.08400130,1.26090932,6.43614197])
OFF=np.array([-.42818514,.05494344]); ROT=-1.93970335
def run(seed,target='box0'):
 e=make_env();s,_=e.reset(seed=seed)
 def g(n,f):return float(s.get(s.get_object_from_name(n),f))
 for k in range(60):
  a=np.zeros(11,np.float32);tx,ty=g(target,'pose_x'),g(target,'pose_y')
  a[0]=np.clip(tx+OFF[0]-g('robot','pos_base_x'),-.2,.2);a[1]=np.clip(ty+OFF[1]-g('robot','pos_base_y'),-.2,.2)
  a[2]=np.clip(ROT-g('robot','pos_base_rot'),-.2,.2)
  for j in range(7):a[3+j]=np.clip(Q[j]-g('robot',f'joint_{j+1}'),-.2,.2)
  a[10]=1;s,*_=e.step(a)
 for k in range(5):
  a=np.zeros(11,np.float32);a[10]=-1;s,*_=e.step(a)
 hit=g('robot','grasp_active')>.5
 print(seed,target,'hit',hit,'baseoff',g('robot','pos_base_x')-g(target,'pose_x'),g('robot','pos_base_y')-g(target,'pose_y'),'q',[round(g('robot',f'joint_{i}'),3) for i in range(1,8)])
 e.close();return hit
if __name__=='__main__':
 for z in range(6):run(z)
