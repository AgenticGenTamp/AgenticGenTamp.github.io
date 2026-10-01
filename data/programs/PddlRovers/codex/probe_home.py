import numpy as np
from env_client import make_env

def one(kind, amount):
 env=make_env(); s,info=env.reset(seed=9)
 rt=next(t for t in env.observation_space.types if t.name=='rover'); r=list(s.get_objects(rt))[0]
 left=amount
 while left > 1e-7:
  a=np.zeros(8,np.float32); part=min(left,.2 if kind=='x' else .4); a[0 if kind=='x' else 2]=part
  s,rew,term,trunc,info=env.step(a); left-=part
 print(kind,amount,'xyt',*(round(float(s.get(r,f)),4) for f in ('x','y','theta')),'home',float(s.get(r,'at_home')))
 env.close()
for amount in (.249,.25): one('x',amount)
for amount in (.38,.39,.392,.3926,.3927,.393,.399,.4): one('theta',amount)
