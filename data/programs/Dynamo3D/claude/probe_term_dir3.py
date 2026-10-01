import sys, numpy as np
from env_client import make_env
from probe_term_lib import *
seed=int(sys.argv[1]); ks=[int(x) for x in sys.argv[2].split(',')]; ndir=int(sys.argv[3]); push=0.03
R=1.0
for k in ks:
    theta=2*np.pi*k/ndir
    env=make_env(); obs,info=env.reset(seed=seed)
    cn=chairs(obs)[0]; c0=cxy(obs,cn); r0=rxy(obs)
    sa=np.arctan2(*(r0-c0)[::-1])
    obs,term,_=move_to(env,obs,c0+R*np.array([np.cos(sa),np.sin(sa)]))
    obs,term=orbit_to(env,obs,c0,theta+np.pi,R)
    if term: print(k,'term positioning',flush=True); env.close(); continue
    d=np.array([np.cos(theta),np.sin(theta)])
    for i in range(300):
        cprev=cxy(obs,cn); rprev=rxy(obs)
        obs,rew,term,trunc,info=env.step(act(d[0]*push,d[1]*push))
        if term or trunc: break
    cf=cfull(obs,cn); ce=np.array([cf['x'],cf['y']]); re=rxy(obs)
    yaw=np.degrees(2*np.arctan2(cf['qz'],abs(cf['qw'])))
    print(f"s{seed} th={np.degrees(theta):6.1f} term={term} c0=({c0[0]:.3f},{c0[1]:.3f}) cprev=({cprev[0]:.3f},{cprev[1]:.3f}) cend=({ce[0]:.3f},{ce[1]:.3f}) rprev=({rprev[0]:.3f},{rprev[1]:.3f}) rend=({re[0]:.3f},{re[1]:.3f}) yaw={yaw:.1f} z={cf['z']:.3f}",flush=True)
    env.close()
