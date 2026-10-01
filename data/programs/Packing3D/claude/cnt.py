from env_client import make_env
env=make_env()
from collections import Counter
c=Counter()
for s in range(40):
    obs,info=env.reset(seed=s)
    c[info.get('object_count')]+=1
print(sorted(c.items()))
try:
    obs,info=env.reset(seed=1,options={'object_count':5})
    print("options ok",info)
except Exception as e:
    print("options fail",e)
env.close()
