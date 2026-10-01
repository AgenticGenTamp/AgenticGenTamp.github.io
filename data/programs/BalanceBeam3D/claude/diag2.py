import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach
import approach as A
seed=int(sys.argv[1]); lo=int(sys.argv[2]); hi=int(sys.argv[3]); bi=int(sys.argv[4])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs,info)
np.set_printoptions(precision=4,suppress=True)
print("blk",obs[bi:bi+7])
for t in range(hi):
    a=ap.get_action(obs)
    obs,r,te,tr,_=env.step(a)
    if lo<=t<hi and t%4==0:
        Rw=A.Rz(obs[18])@A.fk(obs[19:26],A.TOOL)[:3,:3]
        fy=np.arctan2(Rw[1,0],Rw[0,0])
        print(t,"ee",np.round(A.ee_world(obs),4),"fy",round(float(fy),3),"grip",obs[26],"blk",np.round(obs[bi:bi+3],4),flush=True)
    if te: break
env.close()
