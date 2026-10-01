import sys; sys.path.insert(0,"/sandbox")
import numpy as np
from env_client import make_env
env=make_env()
obs,info=env.reset(seed=0, options={'object_count':3})
st=env.get_state()
v=np.asarray(st.vectorize(st) if hasattr(st,'vectorize') else st)
print(type(st))
try:
    print('len',len(v),v[:20])
except Exception as e: print(e)
