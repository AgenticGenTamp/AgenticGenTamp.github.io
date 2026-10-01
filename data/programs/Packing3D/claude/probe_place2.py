import numpy as np
from plib3 import *
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=0); base=rb(obs)
obs,g,m=grasp_part(env,obs,'part0',Rdown,base); print("grasp part0",g,m)
for pz in [0.10,0.095,0.09,0.08]:
    obs,m=place_part(env,obs,'part0',[0.30,0.0,pz],Rdown,base)
    obs2=grip(env,obs,1.0,n=1); ga=rfeat(obs2,'grasp_active'); obs=obs2
    print("release try pz",pz,m,"part",np.round(ppos(obs,'part0'),4),"grasp",ga,flush=True)
    if ga<0.5: break
# xy region for release: re-grasp and try offsets
res={}
for xy in [(0.30,0.0),(0.38,0.0),(0.42,0.0),(0.30,0.14),(0.30,0.18),(0.22,0.0),(0.18,0.0),(0.30,-0.14),(0.45,0.0)]:
    obs,g,m=grasp_part(env,obs,'part0',Rdown,base)
    if not g: print("regrasp fail at",np.round(ppos(obs,'part0'),3),m,flush=True); break
    obs,m=place_part(env,obs,'part0',[xy[0],xy[1],0.10],Rdown,base)
    obs=grip(env,obs,1.0,n=1); ga=rfeat(obs,'grasp_active')
    res[xy]=(m,ga)
    print("xy",xy,m,"grasp_after_open",ga,"part",np.round(ppos(obs,'part0'),3),flush=True)
    if ga>0.5:
        # still holding; nothing to redo
        pass
print(res)
env.close()
