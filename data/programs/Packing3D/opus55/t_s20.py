import numpy as np
from env_client import make_env
from approach import GeneratedApproach, tcp, pose_mat
from ik10 import solve
env=make_env(); obs,info=env.reset(seed=20)
ap=GeneratedApproach(env.action_space, env.observation_space,{}); ap.reset(obs,info)
for t in range(5):
    a=ap.get_action(obs); obs,*_=env.step(a)
c0,ga,gtf=ap._robot(obs); print('ga',ga, 'gtf', np.round(gtf,3))
def trial(dp, use_base, label):
    o,_=env.reset(seed=20)
    ap2=GeneratedApproach(env.action_space, env.observation_space,{}); ap2.reset(o,{})
    for t in range(5):
        a=ap2.get_action(o); o,*_=env.step(a)
    c=ap2._robot(o)[0]; M=tcp(c)
    cn,md,err=solve(c,M[:3,3]+np.array(dp),M[:3,:3],use_base=use_base)
    a=np.zeros(11,np.float32); a[:10]=np.clip(cn-c,-.2,.2); o2,*_=env.step(a)
    print(label, 'md',round(md,3), 'moved', not np.allclose(c, ap2._robot(o2)[0]), np.round(cn-c,3))
trial([0,0,0.01],False,'up1cm arm')
trial([0,0,0.001],False,'up1mm arm')
trial([0,0,-0.001],False,'down1mm arm')
trial([-0.01,0,0],False,'-x1cm arm')
trial([0.0,-0.01,0],False,'-y1cm arm')
trial([-0.01,0,0.01],False,'-x up arm')
trial([0,0,0.07],False,'up7 arm')
trial([0,0,0.07],True,'up7 base')
trial([0,0,0.04],True,'up4 base')
trial([0.0,0.0,0.0],True,'zero base?')
