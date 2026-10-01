import numpy as np
import kinova, ctrl
from env_client import make_env
from runner import Runner
np.set_printoptions(precision=3, suppress=True)
env=make_env(); R=Runner(env,0)
blk=R.obs[54:57].copy()
R.drive(blk[0]-0.45, blk[1], 0.0)
goal=np.array([R.obs[16]+0.35, R.obs[17], 0.30])
Rd=ctrl.Rdown
for i in range(200):
    cur=ctrl.ee_world(R.obs)
    d=goal-R.setp; n=np.linalg.norm(d)
    R.setp = R.setp+d/n*0.015 if n>0.015 else goal.copy()
    dq=kinova.ik_step(R.obs[19:26], ctrl.w2a(R.setp,R.obs[16:19]), Rd, tool_offset=ctrl.TOOL)
    a=np.zeros(11); a[3:10]=np.clip(dq/ctrl.ARM_GAIN,-0.1,0.1)
    R.step(a)
    if i%10==0: print(i,"ee",cur,"setp",R.setp,"dq",np.round(dq,3),"q",np.round(R.obs[19:26],2))
env.close()
