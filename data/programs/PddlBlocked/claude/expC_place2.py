import numpy as np, time, json, sys, fk
from expC_lib import *
B2=np.array([3.72,-0.30,0.0])
def trial(dx,dy,z,carry=None,tag=""):
    t0=time.time(); env=make_env(); log=[]
    obs,q_g,blk=grasp_blocker(env,lambda s: log.append(s))
    rec=dict(tag=tag,dx=dx,dy=dy,z=z,ga0=int(robot(obs)[11]))
    if rec['ga0']!=1: rec['fail']='nograsp'; env.close(); return rec
    n=0; cz = carry if carry is not None else z
    obs,rj,k=move_tool(env,obs,BASE,tool(obs,BASE)+np.array([0,0,cz-0.80]),R0); n+=k; rej=rj
    obs,rj,k=step_to(env,obs,B2,robot(obs)[3:10]); n+=k; rej=rej or rj
    rec['rej_base']=int(rj)
    tp=np.array([4.47+dx,-0.30+dy,cz])
    obs,rj,k=move_tool(env,obs,B2,tp,R0); n+=k; rec['rej_carry']=int(rj)
    if cz!=z:
        obs,rj,k=move_tool(env,obs,B2,np.array([4.47+dx,-0.30+dy,z]),R0); n+=k; rec['rej_down']=int(rj)
    rec['tool']=np.round(tool(obs),3).tolist(); rec['ga_pre']=int(robot(obs)[11])
    rec['blk_held']=np.round(opos(obs,'blocker'),3).tolist()
    a=np.zeros(11,dtype=np.float32); a[10]=1.0
    obs,rew,term,trunc,_=env.step(a); n+=1
    rec['steps_grasp2rel']=n; rec['term']=bool(term); rec['rew']=float(rew)
    rec['ga_after']=int(robot(obs)[11])
    for i in range(3):
        obs,rew,term,trunc,_=env.step(np.zeros(11,dtype=np.float32))
        if term: break
    rec['blk_final']=np.round(opos(obs,'blocker'),3).tolist()
    rec['term_after']=bool(term); rec['t']=round(time.time()-t0,1)
    env.close(); return rec
if __name__=="__main__":
    out=[]
    trials=[(0,0,0.92,None,"base"),(0,0,0.95,None,"high95"),(0,0,1.05,None,"high105"),
            (0,0,0.80,0.92,"low80"),(0,0,0.82,0.92,"low82"),(0,0,0.86,0.92,"low86"),
            (0,0,0.92,0.80,"nolift")]
    for t in trials:
        r=trial(*t); out.append(r); print(json.dumps(r),flush=True)
    json.dump(out,open("expC_r2.json","w"))
