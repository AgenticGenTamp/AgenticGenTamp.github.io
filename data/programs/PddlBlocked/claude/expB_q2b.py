import numpy as np, fk
from env_client import make_env
from lib_util import robot, step_to
from expB_common import *
env=make_env()
obs,s,g,b,d3,yaw=setup(env,1)
R=fk.grasp_R(yaw); base=np.array([3.72,0.10,0.0])
obs,_,_=step_to(env,obs,base,robot(obs)[3:10])
q,_=fk.ik(g-d3*0.25,R,base,robot(obs)[3:10],seeds=6)
obs,_,_=step_to(env,obs,base,q)
for dist in np.arange(0.235,0.184,-0.005):
    qn,e=fk.ik(g-d3*dist,R,base,robot(obs)[3:10],seeds=6)
    obs,rej,n=step_to(env,obs,base,qn,maxsteps=25)
    p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
    print(f"d={dist:.3f} rej={rej} ikerr={e:.4f} achieved={float(np.dot(g-p,d3)):.4f}")
env.close()
