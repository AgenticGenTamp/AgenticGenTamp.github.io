import numpy as np, sys
from env_client import make_env
import approach as A
bname=sys.argv[1]
env=make_env(); obs,info=env.reset(seed=0, options={'object_count':4})
ap=A.GeneratedApproach(None,None,{}); ap.reset(obs,info)
b,q,g=A.robot_state(obs); t=0
while ap.phase!='tobin' and t<250:
    a=ap.get_action(obs); obs,r,te,tr,i=env.step(a); b,q,g=A.robot_state(obs); t+=1
c=A.obj_pos(obs,'cube1')
if c[2]<0.45: print(bname,'GRASP FAILED',np.round(c,3)); sys.exit()
bp=A.obj_pos(obs,bname)
seen={}
for zw in list(np.arange(0.44,0.86,0.03)):
    for k in range(18):
        a,gw=ap._servo(b,q,bp[:2],zw,ap.gclose)
        obs,r,te,tr,i=env.step(a); b,q,g=A.robot_state(obs)
    c=A.obj_pos(obs,'cube1')
    seen[round(float(c[2]),3)]=round(r,4)
print(bname, seen, flush=True)
env.close()
