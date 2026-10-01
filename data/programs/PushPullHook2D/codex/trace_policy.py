from env_client import make_env
from approach import GeneratedApproach
import numpy as np, sys, math
s=int(sys.argv[1]);e=make_env();o,i=e.reset(seed=s);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(o,i);last=-1
for k in range(1000):
 a=p.get_action(o); o,r,t,tr,i=e.step(a)
 if p.phase!=last or k%50==0 or t:
  tip=o[9:11]+o[19]*np.array([np.sin(o[11]),-np.cos(o[11])]);print(k,'ph',p.phase,'rob',np.round(o[:2],3),'hook',np.round(o[9:12],3),'tip',np.round(tip,3),'m',np.round(o[20:22],3),'mc',np.round(o[24:27],2),'t',np.round(o[29:31],3),'tc',np.round(o[33:36],2),'a',np.round(a,3));last=p.phase
 if t or tr:break
e.close()
