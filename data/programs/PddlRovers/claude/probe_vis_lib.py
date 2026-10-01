import numpy as np
from env_client import make_env

OPS = dict(sample=-0.8333, calibrate=-0.5, image=-0.1667, noop=0.1667, send=0.5, drop=0.8333)

def new_env(seed=0, object_count=None):
    env = make_env()
    try:
        obs, info = env.reset(seed=seed, options={'object_count':object_count} if object_count else None)
    except TypeError:
        obs, info = env.reset(seed=seed)
    return env, obs, info

def A(dx=0.,dy=0.,dth=0.,op='noop',i=0, other='noop'):
    a = np.zeros(8, dtype=np.float32)
    a[3]=OPS[other]; a[7]=OPS[other]
    b=4*i
    a[b+0]=dx; a[b+1]=dy; a[b+2]=dth; a[b+3]=OPS[op]
    return a

def st(env, **kw):
    obs,r,te,tr,info = env.step(A(**kw))
    return obs, r, te, tr, info

def pose(obs,i=0):
    o=obs.get_object_from_name(f"rover{i}")
    return np.array([float(obs.get(o,'x')),float(obs.get(o,'y')),float(obs.get(o,'theta'))])

def rf(obs,i=0):
    o=obs.get_object_from_name(f"rover{i}")
    return {f:float(obs.get(o,f)) for f in obs.type_features[o.type]}

def feats(obs,name):
    o=obs.get_object_from_name(name)
    return {f:float(obs.get(o,f)) for f in obs.type_features[o.type]}

def layout(obs):
    d={}
    for n in obs.get_object_names():
        d[n]=feats(obs,n)
    return d

def goto(env, obs, tx, ty, i=0, tol=0.004, maxit=300, other='noop'):
    stuck=0
    for _ in range(maxit):
        p=pose(obs,i)
        ex,ey=tx-p[0],ty-p[1]
        if abs(ex)<tol and abs(ey)<tol: return obs, True
        dx=float(np.clip(ex,-0.2,0.2)); dy=float(np.clip(ey,-0.2,0.2))
        p0=pose(obs,i)
        obs,_,_,_,_=st(env,dx=dx,dy=dy,i=i,other=other)
        if np.linalg.norm(pose(obs,i)[:2]-p0[:2])<1e-7:
            # axis separately
            obs,_,_,_,_=st(env,dx=dx,i=i,other=other)
            obs,_,_,_,_=st(env,dy=dy,i=i,other=other)
            if np.linalg.norm(pose(obs,i)[:2]-p0[:2])<1e-7:
                stuck+=1
                if stuck>2: return obs, False
    return obs, False

def teleport_try(env, obs, tx, ty, i=0):
    """try to reach exactly; returns (obs, ok, pos)"""
    obs, ok = goto(env, obs, tx, ty, i=i)
    return obs, ok, pose(obs,i)

# ---------- navigation ----------
import heapq
def obstacles(obs):
    out=[]
    for n in obs.get_object_names():
        if n.startswith('obstacle'):
            f=feats(obs,n); out.append((n,f['x'],f['y'],f['half_x'],f['half_y'],f['half_z']))
    return out

def build_grid(obs, res=0.05, pad=0.24):
    obs_list=obstacles(obs)
    lo,hi=-2.10,2.10
    n=int((hi-lo)/res)+1
    free=np.ones((n,n),dtype=bool)
    xs=lo+res*np.arange(n)
    X,Y=np.meshgrid(xs,xs,indexing='ij')
    for (nm,x,y,hx,hy,hz) in obs_list:
        free &= ~((np.abs(X-x)<=hx+pad)&(np.abs(Y-y)<=hy+pad))
    # middle wall: x=0, half 0.05, spans y >= -2.5 (impassable everywhere reachable)
    free &= ~(np.abs(X)<=0.05+pad)
    return free,xs,res

def astar(free,xs,res,start,goal):
    n=len(xs)
    def idx(p): return (int(round((p[0]-xs[0])/res)), int(round((p[1]-xs[0])/res)))
    def ok(c): return 0<=c[0]<n and 0<=c[1]<n and free[c[0],c[1]]
    s=idx(start); g=idx(goal)
    if not ok(s):
        # snap to nearest free
        best=None
        for i in range(n):
            for j in range(n):
                if free[i,j]:
                    d=(i-s[0])**2+(j-s[1])**2
                    if best is None or d<best[0]: best=(d,(i,j))
        s=best[1]
    if not ok(g):
        best=None
        for i in range(n):
            for j in range(n):
                if free[i,j]:
                    d=(i-g[0])**2+(j-g[1])**2
                    if best is None or d<best[0]: best=(d,(i,j))
        g=best[1]
    h=lambda c: ((c[0]-g[0])**2+(c[1]-g[1])**2)**0.5
    openq=[(h(s),0,s,None)]; came={}; gs={s:0}
    while openq:
        f,gc,c,par=heapq.heappop(openq)
        if c in came: continue
        came[c]=par
        if c==g: break
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx==0 and dy==0: continue
                nb=(c[0]+dx,c[1]+dy)
                if not ok(nb): continue
                ng=gc+((dx*dx+dy*dy)**0.5)
                if ng<gs.get(nb,1e9):
                    gs[nb]=ng; heapq.heappush(openq,(ng+h(nb),ng,nb,c))
    if g not in came: return None
    path=[]; c=g
    while c is not None:
        path.append((xs[0]+c[0]*res, xs[0]+c[1]*res)); c=came[c]
    return path[::-1]

def nav(env, obs, tx, ty, i=0, tol=0.004, free=None, xs=None, res=None, other='noop'):
    if free is None:
        free,xs,res=build_grid(obs)
    p=pose(obs,i)[:2]
    path=astar(free,xs,res,(p[0],p[1]),(tx,ty))
    if path is None: return obs,False
    # simplify: take every 3rd waypoint
    wps=path[::3]+[(tx,ty)]
    for w in wps:
        obs,ok=goto(env,obs,w[0],w[1],i=i,tol=0.02 if w!=wps[-1] else tol,other=other)
    d=np.linalg.norm(pose(obs,i)[:2]-np.array([tx,ty]))
    return obs, d<max(tol*3,0.02)

def vis_test(env, obs, i=0):
    """Clean visibility oracle: returns (obs, visible) and leaves calibrated=0."""
    obs,_,_,_,_=st(env,op='calibrate',i=i)
    v = rf(obs,i)['calibrated']>0.5
    if v:
        obs,_,_,_,_=st(env,op='image',i=i)
        assert rf(obs,i)['calibrated']<0.5, "image failed to clear"
    return obs, v
