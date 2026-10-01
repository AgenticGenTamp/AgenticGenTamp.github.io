import numpy as np, sys
from env_client import make_env
from probe_lib import *
from probe_solver import rects
env=make_env()
W=3.236
def blk(obs):
    B=obs.get_object_from_name('target_block')
    return {f:obs.get(B,f) for f in ['x','y','theta','width','height','held']}
def run(seed, alpha, g=0.12, L=0.36, off=0.01, push=0.6, verbose=False):
    obs,_=env.reset(seed=seed); b=blk(obs)
    ts=obs.get(obs.get_object_from_name('target_surface'),'x')
    side = 'L' if ts>b['x'] else 'R'
    top=0.1+b['height']
    if side=='L': e=b['x']-b['width']/2; th=-np.pi/2-alpha; sgn=1
    else: e=W-(b['x']+b['width']/2); th=-np.pi/2+alpha; sgn=-1
    # work in mirrored coords: wall at 0, block starting at e. compute in left frame
    thL=-np.pi/2-alpha
    R=rects(0,0,thL,L,g)
    # right finger = the one with lowest y (index 1 = +p)
    V=R[1]; iv=V[:,1].argmin(); cx,cy=V[iv]
    # want corner at (e-off, top+0.01)
    xr=e-off-cx; yr=top+0.01-cy
    allV=np.vstack(R); minx=xr+allV[:,0].min()
    if verbose: print('xr',xr,'yr',yr,'minx',minx)
    if minx<0.002 or xr<0.24: return dict(seed=seed,side=side,e=e,ok=False,why='geom minx %.3f xr %.3f'%(minx,xr))
    X = xr if side=='L' else W-xr
    obs=prep(env,obs)
    obs=seq(env,obs,[None,None,th,L,g],order=(2,3,4))
    obs=seq(env,obs,[X,1.5,None,None,None],order=(1,0))
    obs,_=goto(env,obs,[None,yr+0.05,None,None,None],maxsteps=100)
    b0=blk(obs)
    # descend slowly
    traj=[]
    for i in range(400):
        r0=rob(obs)
        obs,_,_,_,_=step(env,[0,-0.01,0,0,0])
        r1=rob(obs); bb=blk(obs)
        traj.append((r1[1],bb['x'],bb['theta']))
        if r1[1]>r0[1]-0.005: break
    b1=blk(obs)
    # push horizontally away from wall
    for i in range(int(push/0.02)):
        obs,_,_,_,_=step(env,[sgn*0.02,0,0,0,0])
    b2=blk(obs)
    obs,_=goto(env,obs,[None,1.5,None,None,None],maxsteps=40)
    b3=blk(obs)
    ee = (b3['x']-b3['width']/2) if side=='L' else W-(b3['x']+b3['width']/2)
    return dict(seed=seed,side=side,e=round(e,3),h=round(b['height'],3),w=round(b['width'],3),yr_end=round(rob(obs)[1],3),
        dx_desc=round(sgn*(b1['x']-b0['x']),3),th_desc=round(b1['theta'],2),final_gap=round(ee,3),th=round(b3['theta'],2),y=round(b3['y']-b['y'],3))
if __name__=='__main__':
    seed=int(sys.argv[1]); a=float(sys.argv[2]); g=float(sys.argv[3]) if len(sys.argv)>3 else 0.12
    print(run(seed,a,g,verbose=True))
