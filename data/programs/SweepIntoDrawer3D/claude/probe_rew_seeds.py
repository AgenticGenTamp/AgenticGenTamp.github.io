import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True, linewidth=250)
for s in range(8):
    env=make_env(); obs,info=env.reset(seed=s)
    o,r,t,tr,i=env.step(np.zeros(11))
    c=obs[:80].reshape(5,16)
    print("seed",s,"r=%.5f"%r,"term",t,"trunc",tr,"info",i)
    print("  cubes",np.round(c[:,:3],3).tolist())
    print("  bb",np.round(c[0,13:16],3).tolist())
    print("  island",np.round(obs[96:103],3).tolist(),"drawer",np.round(obs[103:109],4).tolist())
    print("  base",np.round(obs[125:128],3).tolist(),"wiper",np.round(obs[147:150],3).tolist())
    print("  unknown 80..96",np.round(obs[80:96],3).tolist())
    print("  unknown 109..125",np.round(obs[109:125],3).tolist())
    env.close()
