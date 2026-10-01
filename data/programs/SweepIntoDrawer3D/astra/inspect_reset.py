from env_client import make_env
import numpy as np
np.set_printoptions(precision=4,suppress=True)
e=make_env()
for seed in [0,1,2]:
 s,i=e.reset(seed=seed)
 print('seed',seed,'info',i)
 print('cubes',s[:80].reshape(5,16)[:,:3]);print('furniture',s[80:125]);print('robot',s[125:147]);print('wiper',s[147:])
 np.save('initial_%s.npy'%seed,s)
e.close()
