import numpy as np, fk, itertools, sys
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

# orientation candidates: assign approach dir a=+x world, up v=+z world to tool axes
a=np.array([1.,0,0]); v=np.array([0,0,1.])
w=np.cross(v,a)
oris={}
oris['x=a,z=v']=np.column_stack([a,w,v])
oris['z=a,x=v']=np.column_stack([v,w,a])
oris['y=a,z=v']=np.column_stack([-w,a,v])
oris['z=a,y=v']=np.column_stack([w,v,a])
oris['x=-a,z=v']=np.column_stack([-a,-w,v])
oris['z=-a,x=v']=np.column_stack([v,-w,-a])
env=make_env()
count=0
for name,R in oris.items():
  for dy in [0.0,-0.19,0.19,-0.38,0.38]:
    for dx in [-0.03,0.05,0.15]:
      if count%40==0:
          obs,_=env.reset(seed=1)
          blk=obs.data[obs.get_object_from_name("blocker")][:3]
          base=np.array([3.72,0.10,0.0]); obs,_=step_to(env,obs,base,robot(obs)[3:10])
      count+=1
      tgt=np.array([blk[0]+dx,blk[1]+dy,blk[2]])
      q_g,e2=fk.ik(tgt,R,base,robot(obs)[3:10],seeds=8,w_rot=0.5)
      if e2>0.02: continue
      obs,rej=step_to(env,obs,base,q_g)
      pe=fk.pose_err(robot(obs)[3:10],base,tgt,R)
      if np.linalg.norm(pe[:3])>0.02 or np.linalg.norm(pe[3:])>0.1: 
          print(f"{name} dy={dy:+.2f} dx={dx:+.2f} unreached {np.round(pe,3)}"); continue
      aa=np.zeros(11,dtype=np.float32); aa[10]=-1.0
      obs,_,_,_,_=env.step(aa); ga=robot(obs)[11]
      print(f"{name} dy={dy:+.2f} dx={dx:+.2f} grasp={ga}")
      if ga>0.5:
          print("   HIT!! joints",np.round(robot(obs)[3:10],4),"base",base)
          print("   block",obs.data[obs.get_object_from_name("blocker")][:3],"tf",np.round(robot(obs)[12:],4))
          sys.exit()
      aa=np.zeros(11,dtype=np.float32); aa[10]=1.0
      obs,_,_,_,_=env.step(aa)
env.close()
