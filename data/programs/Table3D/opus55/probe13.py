from env_client import make_env
from kin import *
import sys
exec(open('probe3.py').read().split('env=make_env()')[0])
seed=int(sys.argv[1]); rng=np.random.default_rng(seed)
def rotz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1.]])
def rotx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
env=make_env(); hits=[]; n=0
for trial in range(60):
    obs,_=env.reset(seed=0); q=getq(obs)
    cx,cy=0.648,-0.173
    off=np.array([rng.uniform(-0.08,0.08),rng.uniform(-0.08,0.08),rng.uniform(0.156,0.30)])
    R=rotz(rng.uniform(-np.pi,np.pi))@rotx(np.pi)@rotx(rng.uniform(-1.0,1.0))@rotz(rng.uniform(-np.pi,np.pi))
    p=np.array([cx+off[0],cy+off[1],0.35])
    qt,e=ik(q,p,R); obs,ok=step_to(env,obs,qt); q=getq(obs)
    qt,e=ik(q,np.array([p[0],p[1],off[2]]),R)
    if e>1e-3: continue
    obs,ok=step_to(env,obs,qt); q=getq(obs)
    if np.linalg.norm(fk(q)[:3,3]-np.array([p[0],p[1],off[2]]))>0.003: continue
    n+=1
    r=obs.get_object_from_name('robot')
    a=np.zeros(11,dtype=np.float32); a[10]=-1
    obs,*_=env.step(a)
    if obs.get(r,'grasp_active')>0: hits.append((off.round(3).tolist(), R.round(2).tolist()))
print('valid',n,'hits',hits)
env.close()
