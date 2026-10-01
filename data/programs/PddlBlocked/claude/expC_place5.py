import numpy as np, json, fk
from expC_place2 import trial
from expC_lib import *
B2=np.array([3.72,-0.30,0.0])
ts=[(0,0.305,0.92,None,"y+0.005"),(0,0.31,0.92,None,"y+0.01"),(0,0.33,0.92,None,"y+0.03"),(0,0.34,0.92,None,"y+0.04"),
    (-0.305,0,0.92,None,"x-0.005"),(-0.31,0,0.92,None,"x-0.01")]
for t in ts:
    r=trial(*t)
    print(json.dumps({k:r[k] for k in ['tag','blk_held','blk_final','ga_after','term'] if k in r}),flush=True)
# recovery test
print("--- recovery ---",flush=True)
env=make_env(); log=[]
obs,q_g,blk=grasp_blocker(env,lambda s: log.append(s)); print(log[0],flush=True)
obs,_,_=move_tool(env,obs,BASE,tool(obs,BASE)+np.array([0,0,0.12]),R0)
obs,_,_=step_to(env,obs,B2,robot(obs)[3:10])
obs,rj,n=move_tool(env,obs,B2,np.array([4.47,-0.28,0.92]),R0)
a=np.zeros(11,dtype=np.float32); a[10]=1.0
obs,rew,term,_,_=env.step(a)
print("open@y+0.02: ga=",robot(obs)[11],"grip=",round(float(robot(obs)[10]),3),"blk=",np.round(opos(obs,'blocker'),3).tolist(),flush=True)
obs,rj,n2=move_tool(env,obs,B2,np.array([4.47,-0.30,0.92]),R0,grip=0.0)
print("after move to centre: ga=",robot(obs)[11],"blk=",np.round(opos(obs,'blocker'),3).tolist(),"steps",n2,flush=True)
obs,rew,term,_,_=env.step(a)
print("open2: ga=",robot(obs)[11],"blk=",np.round(opos(obs,'blocker'),3).tolist(),"term=",term,flush=True)
env.close()
