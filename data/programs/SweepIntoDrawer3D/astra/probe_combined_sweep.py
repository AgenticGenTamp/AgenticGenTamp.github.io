import numpy as np
from env_client import make_env
from kinova import fk,ik,planar_ik
np.set_printoptions(precision=4,suppress=True)
e=make_env();s,_=e.reset(seed=0);h=s[128:135].copy();rot=fk(h)[:3,:3];hi=ik([.7,0,.2],rot,h,max_nfev=100);lo=ik([.7,0,.04],rot,hi,max_nfev=100);w=s[147:150].copy();base=w[:2]+[.9,0];stage=0

def move(q,g,n,target=None,zfeedback=False):
 global s,stage
 q=q.copy();localz=fk(q)[2,3]
 for k in range(n):
  if zfeedback and k%5==0:
   localz=float(np.clip(localz+.5*(.467-s[149]),-.02,.22));q=ik([.7,0,localz],turnrot,q,max_nfev=35)
  a=np.zeros(11,np.float32);err=base-s[125:127] if target is None else np.array(target)-s[147:149];a[:2]=np.clip(err,-.025,.025);a[2]=np.clip(np.pi-s[127],-.1,.1);a[3:10]=np.clip(2*(q-s[128:135]),-.1,.1);a[10]=g;s,r,t,tr,i=e.step(a)
 stage+=1
 print('stage',stage,'wiper',s[147:154],'cube',s[:80].reshape(5,16)[:,:3],'base',s[125:128],'qerr',max(abs(q-s[128:135])), 'drawer',s[107],flush=True);np.save('combined_state_'+str(stage)+'.npy',s)
 return q
drawerq=planar_ik(.6,-.05,pitch=-np.pi/2);drawerq[6]+=np.pi/2
for bx,g,n in [(2.1,0,150),(1.6,0,40),(1.6,1,12),(2.0,1,40),(2.1,0,15)]:
 for k in range(n):
  a=np.zeros(11,np.float32);a[:3]=np.clip(np.array([bx,0,np.pi])-s[125:128],-.03,.03);a[3:10]=np.clip(drawerq-s[128:135],-.1,.1);a[10]=g;s,r,t,tr,i=e.step(a)
 print("DRAWER",bx,g,"joint",s[107],"base",s[125:128],flush=True)
move(hi,0,60);move(lo,0,25);move(lo,1,5);move(hi,1,35);hi[6]+=np.pi;move(hi,1,130)
turnrot=fk(hi)[:3,:3];hi=ik([.7,0,.2],turnrot,hi,max_nfev=100);move(hi,1,65)
y=float(np.mean(s[:80].reshape(5,16)[:,1]));behind=float(np.min(s[:80].reshape(5,16)[:,0])-.15);move(hi,1,50,[behind,y]);lo=ik([.7,0,.1],turnrot,hi,max_nfev=100);lo=move(lo,1,55,[behind,y],True)
for x in np.arange(behind,1.02,.01):lo=move(lo,1,3,[x,y],True)
move(hi,1,35,[1.02,y]);np.save('combined_end.npy',s);e.close()
