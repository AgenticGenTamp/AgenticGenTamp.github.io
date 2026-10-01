from env_client import make_env
import numpy as np
E=make_env(); s,_=E.reset(seed=0)
r=s.get_objects(E.observation_space.get_type('kin_robot'))[0]; t=s.get_objects(E.observation_space.get_type('target_block'))[0]
for i in range(110):
 x,y=s.get(r,'x'),s.get(r,'y');tx,ty=s.get(t,'x'),s.get(t,'y')
 gx,gy=(1.5,3.5) if i<70 else (tx,ty-.5)
 a=np.array([np.clip(gx-x,-.049,.049),np.clip(gy-y,-.049,.049),0,0,0])
 s,rew,done,trunc,info=E.step(a)
 if i%10==0 or done: print(i,'r',s.get(r,'x'),s.get(r,'y'),'t',s.get(t,'x'),s.get(t,'y'),'done',done)
 if done: break
E.close()
