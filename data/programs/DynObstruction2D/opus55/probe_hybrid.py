"""Unwedge a block from a wall. Mirrored coordinates: u = distance from wall-side wall.
Wide blocks (w>0.25): base-circle wedge (arm up, base at wall limit, descend) until gap>=0.45,
then arm-down low push. Narrow blocks: grasp + carry."""
import numpy as np, sys
from env_client import make_env
from probe_lib import *
env=make_env()
W=3.236
def blk(obs):
    B=obs.get_object_from_name('target_block')
    return {f:obs.get(B,f) for f in ['x','y','theta','width','height','held']}
def gapof(b,side):
    return (b['x']-b['width']/2) if side=='L' else W-(b['x']+b['width']/2)
def run(seed, target_gap=0.6, dy=0.005, verbose=False):
    obs,_=env.reset(seed=seed); b=blk(obs)
    ts=obs.get(obs.get_object_from_name('target_surface'),'x')
    side='L' if ts>b['x'] else 'R'; sgn=1 if side=='L' else -1
    X=lambda u: u if side=='L' else W-u          # mirrored -> world x
    e0=gapof(b,side); w=b['width']; h=b['height']; top=0.1+h
    mx=0; method=''
    obs=prep(env,obs)
    if w<=0.25:
        method='grasp'
        uc=max(e0+w/2,0.2401)   # block center (mirrored)
        obs=seq(env,obs,[X(uc),max(1.2,top+0.75),None,None,None],order=(1,0))
        obs=seq(env,obs,[None,None,None,0.48,0.32],order=(3,4))
        tip=max(0.12,top-0.16)
        obs,_=goto(env,obs,[None,tip+0.68,None,None,None],maxsteps=150)
        obs,_=goto(env,obs,[None,None,None,None,0.12],maxsteps=30)
        held=blk(obs)['held']
        obs,_=goto(env,obs,[None,rob(obs)[1]+0.05,None,None,None],maxsteps=30)
        obs,_=goto(env,obs,[X(uc+target_gap+0.05-e0),None,None,None,None],maxsteps=100)
        obs,_=goto(env,obs,[None,rob(obs)[1]-0.04,None,None,None],maxsteps=30)
        obs,_=goto(env,obs,[None,None,None,None,0.32],maxsteps=30)
        obs,_=goto(env,obs,[None,1.2,None,None,None],maxsteps=60)
        method+=' held=%d'%held
    else:
        method='wedge'
        obs=rotate_by(env,obs,np.pi)   # arm up
        obs=seq(env,obs,[X(0.2401),None,None,None,None],order=(0,))
        for i in range(400):
            r0=rob(obs)
            obs,_,_,_,_=step(env,[0,-dy,0,0,0]); r1=rob(obs); bb=blk(obs)
            mx=max(mx,abs(bb['theta']))
            if gapof(bb,side)>=0.45: break
            if r1[1]>r0[1]-dy/2: break
        method+=' ydesc=%.3f gap=%.3f'%(rob(obs)[1],gapof(blk(obs),side))
        # rise, arm down, low push
        obs,_=goto(env,obs,[None,1.2,None,None,None],maxsteps=60)
        obs=seq(env,obs,[X(0.4),None,None,None,None],order=(0,))
        obs=rotate_by(env,obs,-sgn*np.pi)   # arm down, swinging away from wall
        obs=seq(env,obs,[X(0.2401),max(1.2,top+0.75),None,0.48,0.32],order=(0,1,3,4))
        obs,_=goto(env,obs,[None,0.1+0.68+0.02,None,None,None],maxsteps=60)
        e1=gapof(blk(obs),side)
        if verbose: print("lowpose",rob(obs).round(3),e1)
        for i in range(200):
            if verbose and i%5==0: print(rob(obs).round(3),blk(obs))
            if gapof(blk(obs),side)>=target_gap: break
            obs,_,_,_,_=step(env,[sgn*0.02,0,0,0,0]); mx=max(mx,abs(blk(obs)['theta']))
        obs,_=goto(env,obs,[None,1.2,None,None,None],maxsteps=60)
    b2=blk(obs)
    return dict(seed=seed,side=side,e=round(e0,3),h=round(h,3),w=round(w,3),m=method,final_gap=round(gapof(b2,side),3),
                th=round(b2['theta'],2),maxth=round(mx,2),dy=round(b2['y']-b['y'],3))
if __name__=='__main__':
    for s in map(int,sys.argv[1:]): print(run(s),flush=True)
