from probe_vis_lib import *
for seed in [0,1,2,3]:
    env,obs,info=new_env(seed)
    print("== seed",seed,"info",info)
    L=layout(obs)
    for n in sorted(L, key=lambda s:(s.rstrip('0123456789'), int(s.strip('abcdefghijklmnopqrstuvwxyz_') or 0))):
        f=L[n]
        print(n, {k:round(v,3) for k,v in f.items()})
    env.close()
