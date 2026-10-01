import numpy as np, sys, time
from env_client import make_env
from fk import fk
from scipy.optimize import least_squares
J=["joint_%d"%i for i in range(1,8)]
BZ=0.4; HOME=np.array([0,-0.35,-3.1416,-2.5,0,-0.87,1.5708])
def getq(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in J])
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
def ikf(pos,Rt,c0):
    def res(qq):
        T=fk(qq,0.0); e=np.zeros(9); e[:3]=T[:3,3]-pos
        e[3:6]=(T[:3,0]-Rt[:,0])*0.4; e[6:9]=(T[:3,2]-Rt[:,2])*0.4
        return e
    best=None
    for s0 in [c0,HOME]+[HOME+rng.normal(0,1.2,7) for _ in range(3)]:
        s=least_squares(res,s0,xtol=1e-11,ftol=1e-11,max_nfev=150); e=np.linalg.norm(s.fun)
        if best is None or e<best[0]: best=(e,s.x)
        if e<1e-5: break
    return best[1],best[0]
env=make_env(); obs,_=env.reset(seed=0); q=getq(obs); qhome=q.copy()
c=obs.get_object_from_name("cube0")
cp=np.array([float(obs.get(c,f)) for f in ["pose_x","pose_y","pose_z"]])-np.array([0,0,BZ])
def snapshot(o):
    v=[]
    for n in sorted(o.get_object_names()):
        ob=o.get_object_from_name(n)
        if n=="robot":
            v+= [float(o.get(ob,f)) for f in ["finger_state","grasp_active","grasp_tf_x","grasp_tf_y","grasp_tf_z"]]
        else:
            v+= [float(o.get(ob,f)) for f in ["pose_x","pose_y","pose_z","grasp_active"]]
    return np.array(v)
base_snap=snapshot(obs)
def goto(qt,g=0.0,lim=0.25,maxit=50):
    global q,obs
    rej=0
    for _ in range(maxit):
        d=qt-q
        if np.max(np.abs(d))<4e-3: return True
        a=np.zeros(11); a[3:10]=np.clip(d,-lim,lim); a[10]=g
        o2,*_=env.step(a); qn=getq(o2)
        if np.allclose(qn,q,atol=1e-9):
            rej+=1
            if rej>1: return False
        q=qn; obs=o2
    return False
def rand_R():
    # z axis near down, random tilt up to 70deg, random roll
    tilt=rng.uniform(0,1.2); az=rng.uniform(0,2*np.pi); roll=rng.uniform(0,2*np.pi)
    z=np.array([np.sin(tilt)*np.cos(az),np.sin(tilt)*np.sin(az),-np.cos(tilt)])
    a=np.array([0,0,1.0]); x=np.cross(a,z)
    if np.linalg.norm(x)<1e-6: x=np.array([1.,0,0])
    x/=np.linalg.norm(x)
    # roll about z
    y=np.cross(z,x)
    x2=np.cos(roll)*x+np.sin(roll)*y; y2=np.cross(z,x2)
    return np.stack([x2,y2,z],axis=1)
tried=0; t0=time.time()
while time.time()-t0<600:
    off=np.array([rng.uniform(-0.10,0.10),rng.uniform(-0.10,0.10),rng.uniform(0.18,0.30)])
    Rt=rand_R()
    pos=cp+off
    qt,e=ikf(pos,Rt,q)
    if e>1e-3: continue
    if not goto(qt): goto(qhome); continue
    tried+=1
    for g in [-1.0,1.0,-1.0]:
        a=np.zeros(11); a[10]=g; obs,rw,t,tr,i=env.step(a)
        s=snapshot(obs)
        if not np.allclose(s,base_snap,atol=1e-6) or t:
            print("CHANGE! g",g,"pos",np.round(off,3),"Rt",np.round(Rt,2).tolist())
            print(s-base_snap, "term",t); sys.exit()
    if tried%25==0: print("tried",tried,flush=True)
print("done tried",tried)
