from probe_vis_lib import *
import numpy as np
for seed in range(4):
    env,obs,info=new_env(seed)
    L=layout(obs)
    print("== seed",seed)
    for n,f in L.items():
        if n.startswith('objective') or n.startswith('rover') or n.startswith('sample'):
            print(" ",n, {k:round(v,3) for k,v in f.items()})
    for n,f in L.items():
        if n.startswith('obstacle'):
            print(" ",n, round(f['x'],3),round(f['y'],3),round(f['z'],3),round(f['half_x'],3),round(f['half_z'],3))
    env.close()
