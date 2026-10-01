from env_client import make_env
import numpy as np,math

def run(seed):
 e=make_env();s,i=e.reset(seed=seed)
 r=s.get_object_from_name('robot');o=s.get_object_from_name('target_block')
 g=lambda a,f:float(s.get(a,f))
 def move(x,y,theta=0,gap=.32,n=80):
  nonlocal s
  for j in range(n):
   a=np.array([x-g(r,'x'),y-g(r,'y'),theta-g(r,'theta'),.24-g(r,'arm_joint'),gap-g(r,'finger_gap')])
   if np.max(np.abs(a[[0,1,2,3,4]]))<.001: break
   s,*_=e.step(np.clip(a,np.array([-.0499,-.0499,-.1959,-.0999,-.0199]),[.0499,.0499,.1959,.0999,.0199]))
 x=g(o,'x');y=g(o,'y');w=g(o,'width');h=g(o,'height')
 move(g(r,'x'),1.3);move(x-.6,1.3);move(x-.6,max(y,.36))
 trace=[]
 for j in range(60):
  # advance .01 per step, close starting shortly before fingertip contacts side
  gap=.32 if j<12 else 0
  a=[.01,0,0,0,np.clip(gap-g(r,'finger_gap'),-.0199,.0199)]
  s,*_=e.step(np.array(a));
  if j%10==0 or g(o,'held'):
   trace.append((j,round(g(r,'x')-g(o,'x'),3),round(g(r,'y')-g(o,'y'),3),round(g(o,'theta'),3),round(g(r,'finger_gap'),3),g(o,'held')))
  if g(o,'held'): break
 print(seed,'wh',round(w,3),round(h,3),'trace',trace,flush=True)
 e.close()
if __name__=='__main__':
 for seed in range(11):run(seed)
