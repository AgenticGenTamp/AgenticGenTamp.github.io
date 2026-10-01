import numpy as np, fk
from env_client import make_env
from lib_util import robot, step_to
from expB_common import *
env=make_env()
obs,s,g,b,d3,yaw=setup(env,1)
R=fk.grasp_R(yaw); base=np.array([3.72,0.10,0.0])
obs,_,_=step_to(env,obs,base,robot(obs)[3:10])
q,_=fk.ik(g-d3*0.25,R,base,robot(obs)[3:10],seeds=6)
obs,rej,_=step_to(env,obs,base,q)
best=0.25
for dist in np.arange(0.24,0.019,-0.01):
    tgt=g-d3*dist
    qn,e=fk.ik(tgt,R,base,robot(obs)[3:10],seeds=6)
    if e>0.01:
        print(f"d={dist:.2f} IK fail {e:.4f}"); continue
    obs,rej,n=step_to(env,obs,base,qn,maxsteps=25)
    p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
    ach=float(np.dot(g-p,d3))
    print(f"d={dist:.2f} rejected={rej} steps={n} achieved_dist={ach:.4f} tool={np.round(p,3).tolist()}")
    if ach<best: best=ach
    if rej and ach>dist+0.005:
        break
print("MIN along-approach distance reached:",round(best,4))
# try closing there
obs=grip(env,obs,-1.0)
print("grasp_active at min dist:",robot(obs)[11])
env.close()
