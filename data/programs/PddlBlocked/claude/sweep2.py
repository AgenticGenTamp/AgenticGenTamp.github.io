import numpy as np, fk
from env_client import make_env
def robot(obs): return obs.data[obs.get_object_from_name("robot")]
def step_to(env,obs,base_t,q_t,grip=0,maxsteps=40):
    rej=0
    for k in range(maxsteps):
        r=robot(obs); b=r[:3]; q=r[3:10]
        db=np.array(base_t)-b; db[2]=(db[2]+np.pi)%(2*np.pi)-np.pi
        dq=np.array(q_t)-q
        for i in fk.CONT: dq[i]=(dq[i]+np.pi)%(2*np.pi)-np.pi
        a=np.zeros(11,dtype=np.float32)
        a[:3]=np.clip(db,-0.2,0.2); a[3:10]=np.clip(dq,-0.2,0.2); a[10]=grip
        if np.all(np.abs(a[:10])<1e-4): break
        prev=r.copy(); obs,_,_,_,_=env.step(a)
        if np.allclose(robot(obs)[:10],prev[:10],atol=1e-6):
            rej+=1
            if rej>2: return obs,True
        else: rej=0
    return obs,False
env=make_env()
obs,_=env.reset(seed=1)
blk=obs.data[obs.get_object_from_name("blocker")][:3]
base=np.array([3.72,0.10,0.0])
obs,_=step_to(env,obs,base,robot(obs)[3:10])
R=fk.grasp_R(0.0)
for dz in [0.0,0.05]:
  for dx in np.arange(-0.30,0.41,0.05):
    tgt=np.array([blk[0]+dx,blk[1],blk[2]+dz])
    q_g,e2=fk.ik(tgt,R,base,robot(obs)[3:10],seeds=6)
    if e2>0.02:
        print(f"dz={dz:+.2f} dx={dx:+.2f} IKfail {e2:.3f}"); continue
    obs,rej=step_to(env,obs,base,q_g)
    perr=np.linalg.norm(fk.pose_err(robot(obs)[3:10],base,tgt,R)[:3])
    a=np.zeros(11,dtype=np.float32); a[10]=-1.0
    obs,_,_,_,_=env.step(a); ga=robot(obs)[11]
    print(f"dz={dz:+.2f} dx={dx:+.2f} perr={perr:.3f} rej={rej} grasp={ga}")
    if ga>0.5:
        print("   HIT block", obs.data[obs.get_object_from_name("blocker")][:3], "tf", robot(obs)[12:])
    a=np.zeros(11,dtype=np.float32); a[10]=1.0
    obs,_,_,_,_=env.step(a)
env.close()
