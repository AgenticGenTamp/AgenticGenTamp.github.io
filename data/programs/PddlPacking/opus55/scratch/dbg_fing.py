import sys; sys.path.insert(0,'.')
import numpy as np, math
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
pts,_,_,tool,R=fk_full(c[:3],c[3:])
print("tool",tool,"yaw",math.degrees(math.atan2(R[1,1],R[0,1])))
for n,b in blocks.items(): print(n,b)
for dy in np.radians([0,5,10,15,20,30,40,45]):
    for n,b in blocks.items():
        if np.linalg.norm(np.array(b[:2])-tool[:2])<0.03:
            # exact grasp config rotated
            cc=c.copy(); cc[9]+= dy
            boxes=arm_boxes(*[fk_full(cc[:3],cc[3:])[i] for i in (0,4)][:1], fk_full(cc[:3],cc[3:])[4], False)
            ob=OBB(np.array(b[:3]), np.array([[math.cos(b[3]),-math.sin(b[3]),0],[math.sin(b[3]),math.cos(b[3]),0],[0,0,1]]),(0.035,0.035,0.05))
            print(math.degrees(dy), [obb_overlap(bx,ob,0.0) for bx in boxes[3:]])
print("cfg",np.round(cfg,3))
print("cmd",np.round(c,3))
for w in ap.plan:
    p=fk_full(w[0][:3],w[0][3:])[3]
    print("wp",np.round(w[0],3),w[1],"tool",np.round(p,3), nsteps(cfg,w[0]))
w=ap._world(blocks) if hasattr(ap,'_world') else None
print("has _world", w is not None)
W=World(blocks)
print("path_ok", ap._path_ok(cfg, ap.plan[0][0], W), "cfg_ok cmd", ap._config_ok(c, W), "margin", ap.margin)
print("nsteps", nsteps(cfg, ap.plan[0][0]))
g=ap.plan[0][0]; d=cfg_diff(cfg,g); n=nsteps(cfg,g)
for k in range(1,n+1):
    ck=cfg+d*k/n
    print(k, np.round(ck,3), ap._config_ok(ck,W), ap._config_ok(ck,W,exclude='block2'))
