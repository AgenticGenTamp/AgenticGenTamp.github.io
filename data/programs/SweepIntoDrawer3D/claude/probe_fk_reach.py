import numpy as np
from env_client import make_env
from probe_fk_ctl import servo, ee_world
np.set_printoptions(precision=4,suppress=True,linewidth=250)
env=make_env(); o,_=env.reset(seed=0); o=np.asarray(o,float)
PTS=[]
for x in [0.70,0.80,0.90]:
    for y in [0.0,-0.20,-0.40]:
        PTS.append([x,y,0.48])
PTS+=[[0.75,-0.15,0.60],[0.85,-0.05,0.55],[0.60,-0.10,0.50],[0.55,-0.30,0.50],[0.95,-0.45,0.50],[1.00,-0.20,0.60],[0.65,0.10,0.55]]
tot=0
for p in PTS:
    o,e,u=servo(env,o,p,steps=90,grip=0.0); tot+=u
    ee=ee_world(o)
    print(f"tgt={np.array(p)} ee={np.round(ee,4)} err={e:.4f} u={u} tot={tot} {'OK' if e<0.01 else 'BLOCKED/UNREACH'}")
print("base",np.round(o[125:128],4))
env.close()
