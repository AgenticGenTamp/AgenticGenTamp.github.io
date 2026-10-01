import numpy as np, sys
sys.path.insert(0,'/sandbox/old96'); sys.path.insert(1,'/sandbox')
from env_client import make_env
from approach import GeneratedApproach
np.set_printoptions(precision=5,suppress=True,linewidth=250)
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
hist=[]
for t in range(400):
    a=ap.get_action(obs)
    obs,r,te,tr,info=env.step(a)
    hist.append((t,a[10],obs.copy()))
    if te: break
for t,g,o in hist[-6:]:
    print(t,g,'D',o[16:23],'Dv',o[23:29].round(3),'B',o[0:7],'C',o[32:39], 'dd %.4f dc %.4f'%(np.linalg.norm(o[16:18]-o[0:2]),np.linalg.norm(o[32:34]-o[0:2])))
