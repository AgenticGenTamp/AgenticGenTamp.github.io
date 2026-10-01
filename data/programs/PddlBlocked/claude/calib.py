import numpy as np, fk
from env_client import make_env
def robot(obs): return obs.data[obs.get_object_from_name("robot")]
def step_to(env,obs,base_t,q_t,grip=0,maxsteps=40,tol=1e-4):
    rej=0
    for k in range(maxsteps):
        r=robot(obs); db=np.array(base_t)-r[:3]; db[2]=(db[2]+np.pi)%(2*np.pi)-np.pi
        dq=np.array(q_t)-r[3:10]
        for i in fk.CONT: dq[i]=(dq[i]+np.pi)%(2*np.pi)-np.pi
        a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(db,-0.2,0.2); a[3:10]=np.clip(dq,-0.2,0.2); a[10]=grip
        if np.all(np.abs(a[:10])<tol): break
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
got=False
for dz in [-0.175,-0.15,-0.20,-0.16,-0.19,-0.13,-0.22]:
  for dx in [0.0,-0.03,0.03]:
    tgt=np.array([blk[0]+dx,blk[1],blk[2]+dz])
    q_g,e=fk.ik(tgt,R,base,robot(obs)[3:10],seeds=6)
    obs,rej=step_to(env,obs,base,q_g)
    pe=np.linalg.norm(fk.pose_err(robot(obs)[3:10],base,tgt,R)[:3])
    a=np.zeros(11,dtype=np.float32); a[10]=-1.0
    obs,_,_,_,_=env.step(a); ga=robot(obs)[11]
    print(f"dz={dz} dx={dx} ike={e:.4f} rej={rej} pe={pe:.4f} grasp={ga}")
    if ga>0.5: got=True; break
    a=np.zeros(11,dtype=np.float32); a[10]=1.0; obs,_,_,_,_=env.step(a)
  if got: break
if not got: raise SystemExit("no grasp")
r=robot(obs); print("grasp_tf",np.round(r[12:],5))
rng=np.random.default_rng(0); data=[]; q=r[3:10].copy()
lost=0
for trial in range(200):
    qn=fk.clip_q(q+rng.uniform(-0.3,0.3,7))
    obs,rej=step_to(env,obs,base,qn,maxsteps=4)
    r=robot(obs)
    if r[11]<0.5: print("lost grasp",trial); break
    bp=obs.data[obs.get_object_from_name("blocker")]
    data.append(np.concatenate([r[:10],bp[:7]]))
    q=r[3:10].copy()
np.save("calib_data.npy",np.array(data)); print("collected",len(data))
env.close()
