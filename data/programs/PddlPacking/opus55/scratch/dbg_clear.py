import sys; sys.path.insert(0,'.')
import numpy as np
from env_client import make_env
from approach import *
from kin import fk_full
seed=int(sys.argv[1]); upto=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for i in range(upto):
    a=ap.get_action(obs); obs,*_=env.step(a)
cfg,blocks,holding,held,gtf,gq=ap._parse(obs)
a=ap.get_action(obs)
c=cfg+a[:10]
obs2,*_=env.step(a)
cfg2=ap._parse(obs2)[0]
print("holding",holding,held,"rejected", np.abs(cfg2-cfg).max()<1e-6, "grip", a[10])
pts,_,_,tool,R=fk_full(c[:3],c[3:])
boxes=arm_boxes(pts,R,holding)
w=World({n:b for n,b in blocks.items() if n!=held})
def clr(A,B):
    lo,hi=-0.1,0.1
    if obb_overlap(A,B,0.0):
        # penetration: find negative margin? approximate by shrinking
        return -1
    for _ in range(30):
        m=(lo+hi)/2
        if obb_overlap(A,B,m): hi=m
        else: lo=m
    return lo
names=["upper","fore","palm"]+["g%d"%i for i in range(len(boxes)-3)]
for i,bx in enumerate(boxes):
    print(names[i], "table %.4f"%clr(bx,w.table), " ".join("%s %.4f"%(n,clr(bx,ob)) for n,ob in w.boxes.items()))
if holding:
    po,Rrel=gtf,quat_to_R(gq)
    hc=tool+R@po; hb=OBB(hc,R@Rrel,(0.035,0.035,BLOCK_HZ))
    print("held c",hc.round(3), "table %.4f"%clr(hb,w.table), " ".join("%s %.4f"%(n,clr(hb,ob)) for n,ob in w.boxes.items()))
for n,b in blocks.items(): print(n, np.round(b,3))
