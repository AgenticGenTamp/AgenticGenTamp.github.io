import numpy as np
from plib3 import *
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=0); base=rb(obs)
obs,g,m=grasp_part(env,obs,'part0',Rdown,base); print("grasp part0",g,m,flush=True)
def carry2(T,name='part0',hi=0.30):
    global obs
    p=ppos(obs,name)
    obs,m=cmove(env,obs,name,[p[0],p[1],hi],Rdown,base)
    if m!='ok': return 'lift:'+m
    obs,m=cmove(env,obs,name,[T[0],T[1],hi],Rdown,base)
    if m!='ok': return 'move:'+m
    obs,m=cmove(env,obs,name,T,Rdown,base)
    return 'desc:'+m if m!='ok' else 'ok'
def rel(xy,pz=0.10,name='part0'):
    global obs
    m=carry2([xy[0],xy[1],pz],name)
    obs2=grip(env,obs,1.0,n=1); ga=rfeat(obs2,'grasp_active'); obs=obs2
    print("rel",xy,pz,m,"part",np.round(ppos(obs,name),4),"still_grasp",ga,flush=True)
    return ga<0.5
print("released?",rel((0.30,0.0)))
for xy in [(0.38,0.0),(0.42,0.0),(0.30,0.14),(0.30,0.20),(0.22,0.0),(0.20,0.10),(0.40,0.14),(0.30,-0.14)]:
    obs,g,m=grasp_part(env,obs,'part0',Rdown,base)
    if not g: print("regrasp fail",m,np.round(ppos(obs,'part0'),3),flush=True); break
    rel(xy)
env.close()
