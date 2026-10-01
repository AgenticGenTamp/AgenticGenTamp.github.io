from env_client import make_env
import numpy as np
e=make_env()
for seed in range(300):
 s,_=e.reset(seed=seed); u=np.array([np.cos(s[11]),np.sin(s[11])]);n=np.array([u[1],-u[0]])
 g=s[29:31]-s[20:22];g/=np.linalg.norm(g);al=n@g
 if s[21]>1.7 and abs(g[1])<.1 and .6<al<.8:
  print(seed,round(float(s[21]),3),np.round(g,2),round(float(al),2),round(float(s[11]),2))
e.close()
