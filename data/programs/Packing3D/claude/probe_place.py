import numpy as np
from probe_lib import *
env=make_env(); obs,info=env.reset(seed=0)
base=rb(obs)
def grab(name):
    global obs
    p=ppos(obs,name)
    obs,_,_=goto(env,obs,(p[0]-0.10,p[1],0.38),Rdown,base)
    obs,blk,m=goto(env,obs,(p[0]-0.10,p[1],0.30),Rdown,base)
    obs=grip(env,obs,-1.0,n=1)
    return rfeat(obs,'grasp_active')>0.5
def place(name,tgt,tol=1e-3,maxit=8):
    global obs
    for it in range(maxit):
        pp=ppos(obs,name); err=np.array(tgt)-pp
        if np.linalg.norm(err)<tol: return True
        cur=fkpos(obs)+np.array([base[0],base[1],0.0])
        obs,blk,m=goto(env,obs,tuple(cur+err),Rdown,base,maxsteps=50)
        if blk: return False
    return np.linalg.norm(np.array(tgt)-ppos(obs,name))<0.005
print("grab part0",grab('part0'))
place('part0',[0.30,0.0,0.25])
# fine z release scan
rel=None
for pz in np.arange(0.115,0.085,-0.0025):
    place('part0',[0.30,0.0,float(pz)])
    obs=grip(env,obs,1.0,n=1)
    ga=rfeat(obs,'grasp_active')
    print("pz %.4f actual %s grasp %.1f"%(pz,np.round(ppos(obs,'part0'),4),ga),flush=True)
    if ga<0.5: rel=pz; break
print("release at",rel)
# retreat and check part stays
obs,blk,m=goto(env,obs,(0.20,0.0,0.45),Rdown,base)
print("after retreat part0",np.round(ppos(obs,'part0'),4),"grasp",rfeat(obs,'grasp_active'))
# now part1 (triangle)
print("part1 pos",ppos(obs,'part1'))
print("grab part1",grab('part1'))
place('part1',[0.30,0.0,0.25])
for xy in [(0.30,0.10),(0.30,-0.10),(0.35,0.10)]:
    ok=place('part1',[xy[0],xy[1],0.20])
    ok2=place('part1',[xy[0],xy[1],0.10])
    obs=grip(env,obs,1.0,n=1)
    print("try release at",xy,"ok",ok,ok2,"part1",np.round(ppos(obs,'part1'),4),"grasp",rfeat(obs,'grasp_active'),flush=True)
    if rfeat(obs,'grasp_active')<0.5: break
obs,blk,m=goto(env,obs,(0.15,0.0,0.45),Rdown,base)
obs,r,t,tr,i=env.step(np.zeros(11))
print("terminated",t,"rew",r,"part0",np.round(ppos(obs,'part0'),4),"part1",np.round(ppos(obs,'part1'),4),"grasp",rfeat(obs,'grasp_active'))
env.close()
