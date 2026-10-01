import numpy as np, fk
from expC_lib import *
B2=np.array([3.72,-0.30,0.0])
env=make_env(); log=[]
obs,q_g,blk=grasp_blocker(env,lambda s: log.append(s)); print(log[0],flush=True)
obs,_,_=move_tool(env,obs,BASE,tool(obs,BASE)+np.array([0,0,0.12]),R0)
obs,_,_=step_to(env,obs,B2,robot(obs)[3:10])
obs,rj,n=move_tool(env,obs,B2,np.array([4.47,0.02,0.92]),R0)
print("at y+0.02 tool",np.round(tool(obs),3).tolist(),"blk",np.round(opos(obs,'blocker'),3).tolist(),flush=True)
a=np.zeros(11,dtype=np.float32); a[10]=1.0
for i in range(3):
    obs,rew,term,_,_=env.step(a)
    print(f"open{i}: ga={robot(obs)[11]} grip={robot(obs)[10]:.3f} blk={np.round(opos(obs,'blocker'),3).tolist()}",flush=True)
obs,rj,n2=move_tool(env,obs,B2,np.array([4.47,-0.30,0.92]),R0)
print("moved to centre: ga=",robot(obs)[11],"blk",np.round(opos(obs,'blocker'),3).tolist(),"steps",n2,"rej",rj,flush=True)
obs,rew,term,_,_=env.step(a)
print("open again: ga=",robot(obs)[11],"blk",np.round(opos(obs,'blocker'),3).tolist(),"term",term,flush=True)
env.close()
