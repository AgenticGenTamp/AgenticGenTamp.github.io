from env_client import make_env
import numpy as np
E=make_env(); s,info=E.reset(seed=42)
print('INFO',info)
for n in sorted(s.get_object_names()):
 o=s.get_object_from_name(n)
 print(n,o.type if hasattr(o,'type') else o)
 print([(f,s.get(o,f)) for f in E.observation_space.type_features[o.type]])
E.close()
