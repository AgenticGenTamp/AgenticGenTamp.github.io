import numpy as np
from probe_lib import *
env=make_env(); obs,info=env.reset(seed=0)
base=rb(obs); p=ppos(obs,'part0')
rk=ppos(obs,'rack'); print("rack",rk)
obs,_,_=goto(env,obs,(p[0]-0.10,p[1],0.35),Rdown,base)
obs,_,m=goto(env,obs,(p[0]-0.10,p[1],0.30),Rdown,base)
obs=grip(env,obs,-1.0,n=1); print("grasp",rfeat(obs,'grasp_active'))
tgt=np.array([0.30,0.0,0.30])   # desired part pose, high above rack
def place(tgt,label):
    global obs
    cur=fkpos(obs)+np.array([base[0],base[1],0.0])
    for it in range(6):
        pp=ppos(obs,'part0')
        err=tgt-pp
        if np.linalg.norm(err)<2e-3: break
        cmd=cur+err
        obs,blk,m=goto(env,obs,tuple(cmd),Rdown,base,maxsteps=50)
        cur=fkpos(obs)+np.array([base[0],base[1],0.0])
        if blk: 
            print(label,"blocked it",it,"part",np.round(ppos(obs,'part0'),4)); return False
    print(label,"part",np.round(ppos(obs,'part0'),4),"fkworld",np.round(cur,4))
    return True
place(tgt,"above rack")
for pz in [0.20,0.16,0.14,0.130,0.126,0.125,0.120,0.115,0.110,0.105,0.100]:
    ok=place(np.array([0.30,0.0,pz]),"pz%.3f"%pz)
    obs2=grip(env,obs,1.0,n=1)
    ga=rfeat(obs2,'grasp_active'); obs=obs2
    print("   open-> grasp",ga,"part",np.round(ppos(obs,'part0'),4))
    if ga<0.5:
        print("RELEASED at part z",pz); break
    if not ok: break
obs,r,t,term,i=env.step(np.zeros(11))
print("final grasp",rfeat(obs,'grasp_active'),"part0",np.round(ppos(obs,'part0'),4))
env.close()
