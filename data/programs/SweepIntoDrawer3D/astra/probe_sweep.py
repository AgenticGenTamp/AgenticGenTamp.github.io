import numpy as np
from env_client import make_env
from kinova import fk,ik
np.set_printoptions(precision=4,suppress=True)
e=make_env();s,_=e.reset(seed=0);h=s[128:135].copy();rot=fk(h)[:3,:3];hi=ik([.5,0,.2],rot,h,max_nfev=100);lo=ik([.5,0,.04],rot,hi,max_nfev=100);w=s[147:150].copy();base=w[:2]+[.7,0];stage=0

def move(q,g,n,target=None):
 global s,stage
 for k in range(n):
  a=np.zeros(11,np.float32);err=base-s[125:127] if target is None else np.array(target)-s[147:149];a[:2]=np.clip(err,-.035,.035);a[2]=np.clip(np.pi-s[127],-.1,.1);a[3:10]=np.clip(2*(q-s[128:135]),-.1,.1);a[10]=g;s,r,t,tr,i=e.step(a)
  if k%5==0:np.save('sweep_state_'+str(stage)+'_'+str(k)+'.npy',s)
 stage+=1
 print('stage',stage,'robot',s[125:128],'wiper',s[147:150],'cube',s[:80].reshape(5,16)[:,:3],'drawer',s[103:109],flush=True)
move(hi,0,60);move(lo,0,25);move(lo,1,5);move(hi,1,35)
y=float(np.mean(s[:80].reshape(5,16)[:,1]));move(hi,1,40,[.55,y]);move(lo,1,35,[.55,y]);move(lo,1,55,[1.1,y]);move(hi,1,35,[1.1,y]);np.save('sweep_end.npy',s);e.close()
