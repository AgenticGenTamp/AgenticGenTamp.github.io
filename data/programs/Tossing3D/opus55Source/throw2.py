import numpy as np, sys, json
import throw_test as T
from dbg3 import tgrasp
from common import *
k=T.k
def run(seed, qw, w, V, r, n, wait=25, verbose=True):
    bt=tgrasp(seed, 0.9, depth=0.01)
    T.slow(bt,np.array(qw),1.0,1.0)
    s=T.S['s']; base,q=robot(s); p,Rm=k.fk(base,q); c0=obj(s,'cube_0')
    if verbose: print('windup err', np.abs(q-qw).max().round(3), 'pinch', p.round(3), 'cube dist', np.linalg.norm(c0[:3]-p).round(3))
    traj=[]; w=np.array(w,float)
    for i in range(n+wait):
        if i<n: T.step(bt,None,1.0 if i<r else 0.0, vel=V*w, dq=0.1*np.sign(w))
        else:
            base,q=robot(T.S['s']); T.step(bt,q,0.0)
        s=T.S['s']; traj.append(np.concatenate([obj(s,'cube_0'), robot(s)[1]]))
    traj=np.array(traj)
    return bt, c0, traj
if __name__=='__main__':
    seed=int(sys.argv[1]); qw=json.loads(sys.argv[2]); w=json.loads(sys.argv[3]); V=float(sys.argv[4]); r=int(sys.argv[5]); n=int(sys.argv[6])
    bt,c0,traj=run(seed,qw,w,V,r,n)
    print('base',bt.round(3))
    for i,t in enumerate(traj[:14]): print(i, t[:3].round(2), 'v', t[7:10].round(2), 'q', t[10:].round(2))
    f=traj[-1]
    print('final', f[:3].round(3), 'dist from base', np.hypot(*(f[:2]-bt[:2])).round(3))
