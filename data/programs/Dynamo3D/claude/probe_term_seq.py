import sys, numpy as np
from env_client import make_env
from probe_term_lib import *
mode=sys.argv[1]; seed=int(sys.argv[2])
R=1.0; push=0.03
env=make_env(); obs,info=env.reset(seed=seed)
cn=chairs(obs)[0]; c0=cxy(obs,cn); r0=rxy(obs)
def goto_side(obs,theta,center):
    sa=np.arctan2(*(rxy(obs)-center)[::-1])
    obs,t,_=move_to(env,obs,center+R*np.array([np.cos(sa),np.sin(sa)]))
    if t: return obs,True
    return orbit_to(env,obs,center,theta+np.pi,R)
def pushdir(obs,theta,n):
    d=np.array([np.cos(theta),np.sin(theta)])
    for i in range(n):
        obs,rew,t,tr,info=env.step(act(d[0]*push,d[1]*push))
        if t: return obs,True,i
    return obs,False,n

if mode=='pushback':
    # push +x until chair displaced ~0.2, then retreat, then push -x back
    d=np.array([1.,0.])
    obs,t=goto_side(obs,0.0,c0); print('pos',t)
    for i in range(200):
        obs,rew,t,tr,info=env.step(act(0.03,0.))
        disp=np.linalg.norm(cxy(obs,cn)-c0)
        if t: print('TERM during first push disp',round(disp,3)); break
        if disp>0.20: print('reached disp',round(disp,3),'steps',i); break
    if not t:
        c1=cxy(obs,cn)
        obs,t,_=move_to(env,obs,rxy(obs)-np.array([0.6,0.0]))
        print('retreat term',t)
        for _ in range(30):
            obs,rew,t,tr,info=env.step(act(0,0))
            if t: break
        print('idle term',t,'chair',cxy(obs,cn))
        obs,t=goto_side(obs,np.pi,cxy(obs,cn)); print('reposition term',t)
        if not t:
            for i in range(300):
                obs,rew,t,tr,info=env.step(act(-0.03,0.))
                c=cxy(obs,cn); disp0=np.linalg.norm(c-c0)
                if t:
                    print(f'TERM pushing back: chair=({c[0]:.3f},{c[1]:.3f}) disp_from_reset={disp0:.3f} disp_from_c1={np.linalg.norm(c-c1):.3f} pathlen~{0.20+np.linalg.norm(c-c1):.3f}'); break
            else: print('no term after 300 back steps chair',cxy(obs,cn))
elif mode=='perp':
    # push +x by 0.2, retreat, then push +y
    obs,t=goto_side(obs,0.0,c0)
    for i in range(200):
        obs,rew,t,tr,info=env.step(act(0.03,0.))
        if t or np.linalg.norm(cxy(obs,cn)-c0)>0.20: break
    print('phase1 term',t,'chair',cxy(obs,cn))
    if not t:
        c1=cxy(obs,cn)
        obs,t=goto_side(obs,np.pi/2,c1); print('repos term',t)
        for i in range(300):
            obs,rew,t,tr,info=env.step(act(0.,0.03))
            c=cxy(obs,cn)
            if t:
                print(f'TERM: chair=({c[0]:.3f},{c[1]:.3f}) disp_reset={np.linalg.norm(c-c0):.3f} disp_c1={np.linalg.norm(c-c1):.3f} path={0.20+np.linalg.norm(c-c1):.3f}'); break
elif mode=='drive':
    # drive around avoiding chair, 250 steps, check no termination
    pts=[np.array([-1.5,-1.5]),np.array([-1.5,1.5]),np.array([0.0,2.0]),np.array([2.0,2.0]),np.array([2.5,-1.0]),np.array([0.0,-2.0])]
    for p in pts:
        obs,t,n=move_to(env,obs,p,maxsteps=80)
        print('to',p,'term',t,'robot',rxy(obs),'chair',cxy(obs,cn))
        if t: break
env.close()
