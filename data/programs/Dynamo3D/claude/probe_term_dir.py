import sys, numpy as np
from env_client import make_env
from probe_term_lib import *

seed=int(sys.argv[1]); ndir=int(sys.argv[2]) if len(sys.argv)>2 else 8
R=1.0
for k in range(ndir):
    theta=2*np.pi*k/ndir
    env=make_env(); obs,info=env.reset(seed=seed)
    cn=chairs(obs)[0]
    c0=cxy(obs,cn); r0=rxy(obs)
    # go to standoff circle then to push start point
    start_ang=np.arctan2(*(r0-c0)[::-1])
    obs,term,_=move_to(env,obs,c0+R*np.array([np.cos(start_ang),np.sin(start_ang)]),log=None)
    if term: print(k,'term while positioning A'); env.close(); continue
    obs,term=orbit_to(env,obs,c0,theta+np.pi,R)
    if term: print(k,'term while orbiting'); env.close(); continue
    # push inward toward chair and beyond
    d=np.array([np.cos(theta),np.sin(theta)])
    terminated=False; traj=[]
    for i in range(120):
        obs,rew,terminated,trunc,info=env.step(act(d[0]*0.08,d[1]*0.08))
        traj.append((cxy(obs,cn),rxy(obs)))
        if terminated or trunc: break
    cf=cfull(obs,cn); cend=np.array([cf['x'],cf['y']])
    disp=np.linalg.norm(cend-c0)
    print(f"seed{seed} dir{k} th={np.degrees(theta):6.1f} term={terminated} trunc={trunc} steps={i+1} c0=({c0[0]:.3f},{c0[1]:.3f}) cend=({cend[0]:.3f},{cend[1]:.3f}) disp={disp:.3f} z={cf['z']:.3f} q=({cf['qw']:.3f},{cf['qx']:.3f},{cf['qy']:.3f},{cf['qz']:.3f}) v=({cf['vx']:.2f},{cf['vy']:.2f},{cf['vz']:.2f}) robot=({rxy(obs)[0]:.3f},{rxy(obs)[1]:.3f})",flush=True)
    env.close()
