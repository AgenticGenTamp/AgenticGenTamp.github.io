import sys, numpy as np
from env_client import make_env
from probe_term_lib import *
seed=int(sys.argv[1]); ndir=int(sys.argv[2]); push=float(sys.argv[3])
R=1.0
for k in range(ndir):
    theta=2*np.pi*k/ndir
    env=make_env(); obs,info=env.reset(seed=seed)
    cn=chairs(obs)[0]; c0=cxy(obs,cn); r0=rxy(obs)
    sa=np.arctan2(*(r0-c0)[::-1])
    obs,term,_=move_to(env,obs,c0+R*np.array([np.cos(sa),np.sin(sa)]))
    obs,term=orbit_to(env,obs,c0,theta+np.pi,R)
    if term: print(k,'term positioning'); env.close(); continue
    d=np.array([np.cos(theta),np.sin(theta)]); prev=None
    for i in range(300):
        cprev=cxy(obs,cn)
        obs,rew,term,trunc,info=env.step(act(d[0]*push,d[1]*push))
        if term or trunc: break
    cf=cfull(obs,cn); ce=np.array([cf['x'],cf['y']])
    print(f"s{seed} th={np.degrees(theta):6.1f} term={term} c0=({c0[0]:.3f},{c0[1]:.3f}) prev=({cprev[0]:.3f},{cprev[1]:.3f}) d_prev={np.linalg.norm(cprev-c0):.3f} cend=({ce[0]:.3f},{ce[1]:.3f}) d_end={np.linalg.norm(ce-c0):.3f}",flush=True)
    env.close()
