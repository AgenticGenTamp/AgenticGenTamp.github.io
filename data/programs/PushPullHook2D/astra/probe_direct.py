from env_client import make_env
import numpy as np, math
E=make_env(); s,_=E.reset(seed=0)
for t in range(70):
 theta=math.pi/2
 a=[np.clip(s[20]-s[0],-.05,.05), .05, np.clip((theta-s[2]+math.pi)%(2*math.pi)-math.pi,-math.pi/16,math.pi/16), .1,1]
 old=s.copy();s,r,d,tr,i=E.step(np.array(a,dtype=np.float32))
 if t%5==0: print(t,'robot',s[:9].round(3),'button',s[20:22].round(3),'info',i)
 if d:break
E.close()
