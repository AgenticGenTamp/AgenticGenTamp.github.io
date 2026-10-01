import sys, numpy as np
from env_client import make_env
from approach import *
env = make_env(); obs,info=env.reset(seed=int(sys.argv[1]))
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
B=obs[20:22]; T=obs[29:31]
d=T-B; nh=d/np.linalg.norm(d); phi=np.arctan2(nh[1],nh[0])
print('B',B,'T',T,'phi',np.degrees(phi))
from collections import Counter
for op in ap.grasp_options()[:6]:
    cnt=Counter()
    off,dth=op['off'],op['dth']
    for name,rel in [('SO',0),('SI',np.pi),('LI',np.pi/2),('LO',-np.pi/2)]:
        hth=wrap(phi+rel); u,nn=hook_frame(hth)
        for p in np.arange(0.0,1.25,0.05):
            pa,pb={'SO':(0,p),'SI':(HW,p),'LI':(p,HW),'LO':(p,0)}[name]
            for sg in (0.15,0.06):
                for tag,c in (('st',B-nh*(BR+sg)),('end',T-nh*0.13)):
                    V=c-pa*u-pb*nn
                    rx,ry,rth=ap.robot_for_hook(V[0],V[1],hth,off,dth)
                    if not (XMIN<=rx<=XMAX): cnt[(name,tag,'x')]+=1
                    if not (YMIN<=ry<=YMAX): cnt[(name,tag,'y')]+=1
                    if not hook_corners_ok(V[0],V[1],hth,0.03): cnt[(name,tag,'wall')]+=1
    print(op['key'], dict(cnt))
