import numpy as np, sys
from env_client import make_env
from probe_fk_world import move_world, ee_world
np.set_printoptions(precision=4,suppress=True,linewidth=250)
env=make_env(); o,_=env.reset(seed=0); o=np.asarray(o,float)
C=o[:80].reshape(5,16)[:,:3].copy()
tgt=C[3].copy()   # cube3 (0.739,-0.136)
print("cube3",np.round(tgt,4),"ee0",np.round(ee_world(o),4))
seq=[("above_-y",[tgt[0],tgt[1]-0.10,0.60]),
     ("down",    [tgt[0],tgt[1]-0.10,0.472]),
     ("push+y",  [tgt[0],tgt[1]+0.06,0.472])]
tot=0
for name,p in seq:
    o,je,u,ike=move_world(env,o,p,steps=220); tot+=u
    c=o[:80].reshape(5,16)[:,:3]; d=np.linalg.norm(c-C,axis=1)
    print(f"{name} tgt={np.round(p,3)} pred_ee={np.round(ee_world(o),4)} jerr={je:.4f} ikerr={ike:.4f} u={u} tot={tot} dc={np.round(d,3)} base={np.round(o[125:128],3)}")
c=o[:80].reshape(5,16)[:,:3]
print("cube3 moved from",np.round(C[3],4),"to",np.round(c[3],4))
env.close()
