import numpy as np, sys, json
exec(open('calib1.py').read().split("obs=goto(pre,0,obs,300)")[0])
for tgt in [[0.2,0.0,0.8],[0.3,-0.2,0.7],[0.4,-0.3,0.6]]:
    g,err=kin.ik_arm(to_arm(np.array(tgt),base),Rt,obs[96:103],iters=300)
    obs=goto(g,0,obs,300,0.002)
    print(tgt,'fk',kin.fk_world(obs[93:96],obs[96:103])[0],'qerr',obs[96:103]-g)
