import numpy as np, fk
def robot(obs): return obs.data[obs.get_object_from_name("robot")]
def step_to(env,obs,base_t,q_t,grip=0.0,maxsteps=60,verbose=False):
    rej=0; n=0
    for k in range(maxsteps):
        r=robot(obs); db=np.array(base_t)-r[:3]; db[2]=(db[2]+np.pi)%(2*np.pi)-np.pi
        dq=np.array(q_t)-r[3:10]
        for i in fk.CONT: dq[i]=(dq[i]+np.pi)%(2*np.pi)-np.pi
        a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(db,-0.2,0.2); a[3:10]=np.clip(dq,-0.2,0.2); a[10]=grip
        if np.all(np.abs(a[:10])<1e-4): break
        prev=r.copy(); obs,rew,term,trunc,_=env.step(a); n+=1
        if np.allclose(robot(obs)[:10],prev[:10],atol=1e-6):
            rej+=1
            if rej>2: return obs,True,n
        else: rej=0
    return obs,False,n
