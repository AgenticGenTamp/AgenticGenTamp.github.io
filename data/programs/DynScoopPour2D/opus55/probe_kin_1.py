import numpy as np
from env_client import make_env
env=make_env()
obs,info=env.reset(seed=0)
R=lambda o: {f: round(o.get(o.get_object_from_name('robot'),f),4) for f in ['x','y','theta','arm_joint','arm_length','finger_gap']}
def run(a,n,obs):
    last=None
    for i in range(n):
        obs,r,te,tr,info=env.step(np.array(a,dtype=float))
        cur=R(obs)
        if last is not None and cur==last: break
        last=cur
    return obs,i,r,te,tr
# exactness
o0=R(obs); obs,_,_,_,_=env.step(np.array([0.01,-0.013,0.05,0.03,-0.005])); print('single step',o0,'->',R(obs))
# extremes up
obs,i,r,te,tr=run([0,0.03,0,0,0],200,obs); print('up',i,R(obs),r,te,tr)
obs,i,*_=run([0.03,0,0,0,0],200,obs); print('right',i,R(obs))
obs,i,*_=run([-0.03,0,0,0,0],300,obs); print('left top',i,R(obs))
obs,i,*_=run([0,-0.03,0,0,0],300,obs); print('down at left',i,R(obs))
# theta wrap
obs,i,*_=run([0,0,0.098,0,0],100,obs); print('theta spin',i,R(obs))
obs,i,*_=run([0,0,0,0.08,0],100,obs); print('arm max',i,R(obs))
obs,i,*_=run([0,0,0,-0.08,0],100,obs); print('arm min',i,R(obs))
obs,i,*_=run([0,0,0,0,0.015],100,obs); print('grip max',i,R(obs))
obs,i,*_=run([0,0,0,0,-0.015],100,obs); print('grip min',i,R(obs))
