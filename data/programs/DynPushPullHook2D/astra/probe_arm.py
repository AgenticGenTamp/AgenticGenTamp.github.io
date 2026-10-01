from env_client import make_env
import numpy as np
E=make_env(); s,_=E.reset(seed=0)
r=s.get_objects(E.observation_space.get_type('kin_robot'))[0]; t=s.get_objects(E.observation_space.get_type('target_block'))[0]
for i in range(100):
 if i<24: a=[0,0,.0628,0,-.019]
 elif i<45: a=[0,.049,0,0,0]
 elif i<65: a=[0,0,0,.099,0]
 else: a=[0,0,0,-.099,-.019]
 s,rew,done,trunc,info=E.step(np.array(a))
 if i%5==0 or done:print(i,'r',[round(s.get(r,f),3) for f in ['x','y','theta','arm_length','finger_gap']],'t',[round(s.get(t,f),3) for f in ['x','y','held']],'done',done)
 if done:break
E.close()
