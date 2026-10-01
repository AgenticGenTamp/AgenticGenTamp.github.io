from env_client import make_env
import numpy as np, time
env = make_env()
def q(obs,k):
    R=obs.get_object_from_name('robot'); return float(obs.get(R,k))
def stepa(**kw):
    a=np.zeros(11,np.float32)
    for k,v in kw.items(): a[int(k[1:])]=v
    return env.step(a)
for d in [-0.15,-0.16,-0.17,-0.2]:
    obs,_=env.reset(seed=0); obs,*_=stepa(a6=d); print("j4",d,round(q(obs,'joint_4'),4))
# j6 negative: j2/j4 move? try j6 -0.2 after lifting j4
for d in [-0.33,-0.34,-0.4]:
    obs,_=env.reset(seed=0); obs,*_=stepa(a8=d); print("j6",d,round(q(obs,'joint_6'),4))
t=time.time(); obs,_=env.reset(seed=0)
for i in range(50): obs,*_=stepa(a10=-1)
print("50 steps sec",time.time()-t, "finger",q(obs,'finger_state'), [round(q(obs,k),3) for k in ['grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qw']])
