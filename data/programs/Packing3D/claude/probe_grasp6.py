import numpy as np, json
from probe_lib import *
env=make_env(); obs,info=env.reset(seed=0)
base=rb(obs); p=ppos(obs,'part0')
obs=grip(env,obs,1.0)
obs,_,_=goto(env,obs,(p[0]-0.10,p[1],0.35),Rdown,base)
obs,_,_=goto(env,obs,(p[0]-0.10,p[1],0.30),Rdown,base)
obs=grip(env,obs,-1.0)
assert rfeat(obs,'grasp_active')>0.5
rows=[]
rng=np.random.default_rng(0)
q0=getq(obs)
for k in range(25):
    qd=q0+rng.uniform(-0.35,0.35,7)
    obs,blk=move_to(env,obs,qd)
    q=getq(obs)
    o=obs.get_object_from_name('part0')
    pp=[obs.get(o,f) for f in ['pose_x','pose_y','pose_z','pose_qx','pose_qy','pose_qz','pose_qw']]
    rows.append({'q':q.tolist(),'part':pp,'blk':bool(blk)})
json.dump({'base':base.tolist(),'rows':rows},open('calib.json','w'))
print("collected",len(rows),"nblocked",sum(r['blk'] for r in rows))
env.close()
