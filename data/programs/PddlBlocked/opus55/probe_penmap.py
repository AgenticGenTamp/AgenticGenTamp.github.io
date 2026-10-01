from env_client import make_env
import numpy as np, argparse
from kin import ik, fk_world
from envutil import Sim
ap=argparse.ArgumentParser()
ap.add_argument('seed',type=int); ap.add_argument('--obj',default='green0')
ap.add_argument('--app',default='d')   # 'd' or angle in deg (world)
ap.add_argument('--frame',default='d') # 'd' or 'app'
ap.add_argument('--base',default=None) # x,y,yaw
ap.add_argument('--u',default='-0.2:0.2:0.04'); ap.add_argument('--v',default='-0.2:0.2:0.04')
ap.add_argument('--zt',type=float,default=1.10); ap.add_argument('--em',type=float,default=None); ap.add_argument('--origin',default=None); ap.add_argument('--grip',type=float,default=0); ap.add_argument('--rm',type=int,default=0); ap.add_argument('--via',default=''); ap.add_argument('--tilt',type=float,default=0); ap.add_argument('--zmin',type=float,default=0.805)
A=ap.parse_args()
def rng(s):
    if ':' in s:
        a,b,c=[float(x) for x in s.split(':')]; return [float(x) for x in np.round(np.arange(a,b+1e-6,c),3)]
    return [float(x) for x in s.split(',')]
us=rng(A.u); vs=rng(A.v)
env=make_env()
fresh=lambda: Sim(env,env.reset(seed=A.seed)[0])
S=fresh(); g=S.block(A.obj); bl=S.block('blocker'); g0=S.block('green0')
if A.origin: g=np.r_[[float(x) for x in A.origin.split(',')],0.8]
d=np.r_[g0[:2]-bl[:2],0]; d/=np.linalg.norm(d)
app=d if A.app=='d' else -d if A.app=='md' else np.array([np.cos(np.radians(float(A.app))),np.sin(np.radians(float(A.app))),0])
fd=d if A.frame=='d' else app
th=np.radians(A.tilt); h=app.copy(); app=np.r_[np.cos(th)*h[:2],-np.sin(th)]; ZAX=tuple(np.r_[np.sin(th)*h[:2],np.cos(th)])
perp=np.array([-fd[1],fd[0],0])
BASE=np.array([float(x) for x in A.base.split(',')]) if A.base else np.array([3.75,g[1]-0.17,0.0])
q_start=S.q(); ZT=A.zt
def setup(p):
    global S
    S=fresh()
    if A.rm:
        from probe_pm_rm import remove_blocker
        remove_blocker(S)
        if A.via: print('tuck',S.moveto(S.base(),q_start))
        for w in [x for x in A.via.split(';') if x]: print('via',w,S.moveto([float(t) for t in w.split(',')],q_start),S.steps)
        S.moveto(BASE,q_start if A.via else S.q())
    else: S.moveto(np.r_[BASE[:2],BASE[2]],q_start)
    _,q,e=ik(np.r_[p[:2],ZT],app,BASE,S.q(),free_base=False,elbow_min=A.em,zaxis=ZAX)
    ok=S.moveto(BASE,q,maxd=0.1)
    for _ in range(3): S.step(np.r_[np.zeros(10),A.grip])
    return ok
res={}; need=True
for u in us:
  for v in vs:
    p=g[:3]+u*fd+v*perp
    if need or S.steps>900:
        if not setup(p): print('setup fail',u,v); need=True; continue
        need=False
    _,q,e=ik(np.r_[p[:2],ZT],app,BASE,S.q(),free_base=False,elbow_min=A.em,zaxis=ZAX)
    if e>3e-3: res[(u,v)]=None; continue
    if not S.moveto(BASE,q,maxd=0.05):
        res[(u,v)]=9; need=True; continue
    path=[q]; low=ZT; z=ZT
    while z>A.zmin:
        z=round(z-0.01,3)
        _,qq,e=ik(np.r_[p[:2],z],app,BASE,S.q(),free_base=False,elbow_min=A.em,zaxis=ZAX)
        if e>3e-3: low=-1; break
        if not S.moveto(BASE,qq,maxd=0.05): break
        path.append(qq); low=z
    res[(u,v)]=low
    for qq in path[::-1]:
        if not S.moveto(BASE,qq,maxd=0.1): need=True; break
print('seed',A.seed,'obj',A.obj,'app',app[:2].round(3),'frame u-dir',fd[:2].round(3),'base',BASE)
print('lowest tool z. rows u, cols v:', ' '.join('%5.2f'%v for v in vs))
for u in us[::-1]:
    print('%5.2f'%u, ' '.join('  -- ' if res.get((u,v)) is None else ' XXX ' if res[(u,v)]==9 else ' IK  ' if res[(u,v)]==-1 else '%5.2f'%res[(u,v)] for v in vs))
