import numpy as np
from env_client import make_env
np.set_printoptions(precision=4,suppress=True,linewidth=250)
env=make_env(); o,_=env.reset(seed=0); o=np.asarray(o,float)
print("cubes:\n",o[:80].reshape(5,16)[:,:3])
print("island pose",o[96:103])
print("drawers",o[103:109])
print("base",o[125:128])
print("joints",o[128:135],"grip",o[135])
print("wiper",o[147:153],"bb",o[160:163])
np.save('obs0_fk.npy',o)
env.close()
