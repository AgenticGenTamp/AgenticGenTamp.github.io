import numpy as np, sys, fk
from expC_lib import *
env=make_env(); L=lambda s: print(s,flush=True)
obs,q_g,blk=grasp_blocker(env,L)
r=robot(obs); L(f"tool={np.round(tool(obs,BASE),3).tolist()} ga={r[11]} grip_open={r[10]:.3f} blk={np.round(opos(obs,'blocker'),3).tolist()}")
TOT=0
# lift
for dz in [0.12]:
    tg=tool(obs,BASE)+np.array([0,0,dz])
    obs,rj,n=move_tool(env,obs,BASE,tg,R0); TOT+=n
    L(f"lift{dz}: steps={n} rej={rj} tool={np.round(tool(obs,BASE),3).tolist()} ga={robot(obs)[11]} blk={np.round(opos(obs,'blocker'),3).tolist()}")
B2=np.array([3.72,-0.30,0.0])
obs,rj,n=step_to(env,obs,B2,robot(obs)[3:10]); TOT+=n
L(f"basemove: steps={n} rej={rj} base={np.round(robot(obs)[:3],3).tolist()} ga={robot(obs)[11]} blk={np.round(opos(obs,'blocker'),3).tolist()} tool={np.round(tool(obs),3).tolist()}")
tp=np.array([4.47,-0.30,0.92])
obs,rj,n=move_tool(env,obs,B2,tp,R0); TOT+=n
L(f"toplate: steps={n} rej={rj} tool={np.round(tool(obs),3).tolist()} ga={robot(obs)[11]} blk={np.round(opos(obs,'blocker'),3).tolist()}")
a=np.zeros(11,dtype=np.float32); a[10]=1.0
obs,rew,term,trunc,_=env.step(a); TOT+=1
L(f"open: rew={rew} term={term} ga={robot(obs)[11]} grip={robot(obs)[10]:.3f} blk={np.round(opos(obs,'blocker'),3).tolist()} TOTsteps_grasp2rel={TOT}")
for i in range(6):
    a=np.zeros(11,dtype=np.float32)
    obs,rew,term,trunc,_=env.step(a)
    L(f"  settle{i}: term={term} ga={robot(obs)[11]} blk={np.round(opos(obs,'blocker'),3).tolist()}")
    if term: break
env.close()
