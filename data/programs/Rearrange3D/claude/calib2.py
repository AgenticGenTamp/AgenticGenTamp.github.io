import numpy as np, json, sys
from env_client import make_env
import kinova_fk as K

MOUNT=np.array([0.0,0.0,0.44])
def R_down(yaw=0.0):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])@np.array([[1,0,0],[0,-1,0],[0,0,-1.]])

def world_to_rel(p, base, mount=MOUNT):
    bx,by,th=base
    v=np.array([p[0]-bx,p[1]-by]); c,s=np.cos(th),np.sin(th)
    r=np.array([c*v[0]+s*v[1], -s*v[0]+c*v[1]])
    return np.array([r[0]-mount[0], r[1]-mount[1], p[2]-mount[2]])

def step_to(env,obs,qt,steps,grip,kp=3.0):
    for i in range(steps):
        a=np.zeros(11,dtype=np.float32); a[10]=grip
        q=np.asarray(obs)[96:103]
        a[3:10]=np.clip(kp*(qt-q),-0.1,0.1)
        obs,r,te,tr,inf=env.step(a)
    return obs

def move_to(env,obs,p,steps=8,grip=0.0,yaw=0.0,maxjump=1.5):
    q=np.asarray(obs)[96:103].copy()
    qt,ok,inf=K.ik(np.asarray(p),R_down(yaw),q_init=q)
    if not ok or inf['pos_err']>0.005: return obs,False
    if np.max(np.abs(qt-q))>maxjump: return obs,False
    return step_to(env,obs,qt,steps,grip),True

def move_line(env,obs,p0,p1,n=10,steps_per=6,grip=0.0,yaw=0.0,rec=None):
    for t in np.linspace(0,1,n+1):
        p=(1-t)*np.asarray(p0,float)+t*np.asarray(p1,float)
        obs,ok=move_to(env,obs,p,steps_per,grip,yaw)
        if rec is not None: rec.append((p.copy(),np.asarray(obs).copy(),ok))
    return obs

def sweep(axis='x', z=0.05, seed=0, other=None, span=(0.45,0.85), gr=0.0):
    env=make_env(); obs,info=env.reset(seed=seed)
    o=np.asarray(obs); base=o[93:96].copy()
    print("bowl",np.round(o[0:3],3),"drink",np.round(o[16:19],3),"can",np.round(o[32:35],3),"base",np.round(base,3))
    relb=world_to_rel(o[0:3],base); print("bowl rel",np.round(relb,3))
    idx=[0,1,16,17,32,33]; ref=o[idx].copy()
    if axis=='x':
        p0=np.array([span[0], relb[1] if other is None else other, z]); p1=np.array([span[1],p0[1],z])
    else:
        p0=np.array([relb[0] if other is None else other, span[0], z]); p1=np.array([p0[0],span[1],z])
    # approach from above
    obs,ok=move_to(env,obs,p0+np.array([0,0,0.25]),steps=90,grip=gr,maxjump=99)
    print("prepose ok",ok,np.round(K.fk(np.asarray(obs)[96:103])[:3,3],3))
    obs=move_line(env,obs,p0+np.array([0,0,0.25]),p0,n=3,steps_per=10,grip=gr)
    rec=[]
    obs=move_line(env,obs,p0,p1,n=int(round((span[1]-span[0])/0.025)),steps_per=5,grip=gr,rec=rec)
    for p,ob,ok in rec:
        d=ob[idx]-ref
        fk=K.fk(ob[96:103])[:3,3]
        print("cmd %s=%.3f ok=%d fk=(%.3f,%.3f,%.3f) del=%s"%(axis,p[0] if axis=='x' else p[1],ok,fk[0],fk[1],fk[2],np.round(d,3)))
    env.close()

if __name__=="__main__":
    sweep(axis=sys.argv[1] if len(sys.argv)>1 else 'x')
