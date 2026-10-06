import numpy as np, time
from test_approach import make
from approach import *
env=make(); s,info=env.reset(seed=10, options={'object_count':30})
ap=GeneratedApproach(env.action_space, None, {}); ap.reset(s,info)
for t in range(300):
    if not ap.plan and t>0:
        r,rects=ap._parse(s); m=ap._robot_model(r)
        if r['vacuum']<0.5 and np.linalg.norm(ap._q(r)[:2]-[0.82,0.18])<0.05:
            break
    a=ap.get_action(s); s,_,_,_,_=env.step(a)
q0=ap._q(r); print("q0",q0)
chk0=Checker(m,ap._obstacles(rects,set()),[])
print("free margin",chk0.free(q0),"free0",chk0.free(q0,0.0))
h,ok=chk0.colliders(q0,margin=0.003); print("colliders",[chk0.obs.names[i] for i in np.nonzero(h)[0]],ok)
# try random small moves
cnt=0
for _ in range(200):
    q=q0+np.random.uniform(-1,1,4)*STEP; q[3]=np.clip(q[3],0.1,0.2)
    cnt+=chk0.free(q)
print("free neighbors",cnt)
for n,d in rects.items():
    c=rect_verts(d['x'],d['y'],d['th'],d['w'],d['h']).mean(0)
    if np.linalg.norm(c-q0[:2])<0.4: print(n,np.round(c,2),round(d['w'],2),round(d['h'],2),round(d['th'],2))
rng=np.random.default_rng(0)
for goal in [[1.5,1.5,0,0.1],[0.8,0.6,0,0.1],[2.0,0.5,0,0.1]]:
    g=np.array(goal); print(goal, chk0.free(g))
    t=time.time(); p=rrt_connect(chk0,q0,[g],rng,time_limit=3.0); print(" path", None if p is None else len(p), time.time()-t)
