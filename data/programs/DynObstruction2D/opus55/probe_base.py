import numpy as np, sys
from env_client import make_env
from probe_lib import *
env=make_env()
W=3.236
def blk(obs):
    B=obs.get_object_from_name('target_block')
    return {f:obs.get(B,f) for f in ['x','y','theta','width','height','held']}
def run(seed, dy=0.005, push=0.6, xoff=0.0, verbose=False):
    obs,_=env.reset(seed=seed); b=blk(obs)
    ts=obs.get(obs.get_object_from_name('target_surface'),'x')
    side = 'L' if ts>b['x'] else 'R'; sgn=1 if side=='L' else -1
    e = b['x']-b['width']/2 if side=='L' else W-(b['x']+b['width']/2)
    obs=prep(env,obs)
    obs=seq(env,obs,[None,None,np.pi/2,None,None],order=(2,))
    X=0.24+xoff if side=='L' else W-0.24-xoff
    obs=seq(env,obs,[X,None,None,None,None],order=(0,))
    b0=blk(obs); mx=0
    for i in range(400):
        r0=rob(obs)
        obs,_,_,_,_=step(env,[0,-dy,0,0,0]); r1=rob(obs); bb=blk(obs)
        mx=max(mx,abs(bb['theta']))
        if verbose and i%5==0: print(r1.round(3),'blk',round(bb['x'],3),round(bb['y'],3),round(bb['theta'],3))
        if r1[1]>r0[1]-dy/2: break
    b1=blk(obs); r=rob(obs)
    for i in range(int(push/0.02)):
        obs,_,_,_,_=step(env,[sgn*0.02,0,0,0,0]); bb=blk(obs); mx=max(mx,abs(bb['theta']))
    b2=blk(obs)
    ee = (b2['x']-b2['width']/2) if side=='L' else W-(b2['x']+b2['width']/2)
    return dict(seed=seed,side=side,e=round(e,3),h=round(b['height'],3),w=round(b['width'],3),ybot=round(r[1],3),
        dx_desc=round(sgn*(b1['x']-b0['x']),3),final_gap=round(ee,3),th=round(b2['theta'],2),maxth=round(mx,2),dy=round(b2['y']-b['y'],3))
if __name__=='__main__':
    for s in map(int,sys.argv[1:]): print(run(s))
