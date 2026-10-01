import numpy as np
from env_client import make_env
from approach import GeneratedApproach, tcp, pose_mat
from ik10 import solve
env=make_env(); obs,info=env.reset(seed=1)
ap=GeneratedApproach(env.action_space, env.observation_space,{}); ap.reset(obs,info)
for t in range(3):
    a=ap.get_action(obs); obs,*_=env.step(a)
c,ga,gtf=ap._robot(obs); print('ga',ga)
G=pose_mat(gtf[:3],gtf[3:7]); Gi=np.linalg.inv(G)
def cfg_for(c, xyz):
    P=np.eye(4); P[:3,3]=xyz; T=P@Gi
    cn,md,err=solve(c,T[:3,3],T[:3,:3]); return cn,md
def step_to(cn):
    global obs
    c0=ap._robot(obs)[0]; d=cn-c0
    a=np.zeros(11,np.float32); a[:10]=np.clip(d,-.2,.2); obs,*_=env.step(a)
    c1=ap._robot(obs)[0]; return not np.allclose(c0,c1)
# A: one move from table (0.179,0.314,0.095) to the floor (0.3,0.07,0.098), sliding through the wall
cn,md=cfg_for(c,[0.28,0.2,0.1]); print('md',md, 'moved', step_to(cn))
P=ap._part_info(obs,'part0')['pose'][:3]; print('part',P)
c=ap._robot(obs)[0]
cn,md=cfg_for(c,[0.28,0.07,0.1]); print('md',md, 'moved thru wall', step_to(cn))
P=ap._part_info(obs,'part0')['pose'][:3]; print('part',P)
c=ap._robot(obs)[0]
cn,md=cfg_for(c,[0.28,0.07,0.2]); print('md',md, 'moved up', step_to(cn))
c=ap._robot(obs)[0]
cn,md=cfg_for(c,[0.28,0.07,0.1]); print('md',md, 'moved down', step_to(cn))
P=ap._part_info(obs,'part0')['pose'][:3]; print('part',P)
def go(xyz):
    c=ap._robot(obs)[0]; cn,md=cfg_for(c,xyz); n=int(np.ceil(md/0.19)); ok=True
    for k in range(n):
        c0=ap._robot(obs)[0]; ok=step_to(c0+(cn-c)/n) and ok
    return ok, n
print('up', go([0.28,0.2,0.2])); print('over', go([0.28,0.07,0.2])); print('down', go([0.28,0.07,0.1]))
print('part', ap._part_info(obs,'part0')['pose'][:3])
