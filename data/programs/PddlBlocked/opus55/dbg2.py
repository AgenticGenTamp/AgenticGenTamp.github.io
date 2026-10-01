import sys, numpy as np
from env_client import make_env
from envutil import Sim
import planner
from kin import ik, fk_points, rect_pen
env=make_env(); S=Sim(env,env.reset(seed=int(sys.argv[1]))[0])
blk=S.block('blocker'); g0=S.block('green0'); z=g0[2]
d=np.r_[g0[:2]-blk[:2],0]; d/=np.linalg.norm(d)
cb=np.r_[blk[:2],z]; cg=np.r_[g0[:2],z]
print('d',d.round(3))
for b0 in planner.side_seeds(planner.TABLE,cg):
    b,q,err=ik(cg,d,b0,planner.HIGH_Q,free_base=True,base_nom=b0,elbow_min=0.82,table=planner.TABLE)
    if err>5e-3 or rect_pen(b,0.36,planner.TABLE)>0: continue
    out=[]
    for name,t in [('cb',cb),('cb-.1d',cb-0.1*d),('cb+.1up',cb+[0,0,0.1]),('prehi',cb-0.1*d+[0,0,0.1]),('cg+.1up',cg+[0,0,0.1])]:
        _,qq,e=ik(t,d,b,q,free_base=False,elbow_min=0.82); out.append((name,round(e,3)))
    print('base',b.round(2),'elbow z',fk_points(b,q)[1][2].round(3),out)
