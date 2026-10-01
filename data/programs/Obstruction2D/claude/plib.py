from env_client import make_env
import numpy as np
def d(obs,n):
    o=obs.get_object_from_name(n); return {f: float(obs.get(o,f)) for f in obs.type_features[o.type]}
def step(env,dx=0,dy=0,dth=0,da=0,v=0):
    return env.step(np.array([dx,dy,dth,da,v],dtype=np.float64))
def move_to(env,obs,field,target,idx,lim,vac=0,maxit=200,tol=1e-6):
    """field in robot dict; idx 0=x,1=y,3=arm"""
    for i in range(maxit):
        r=d(obs,'robot'); cur=r[field]; err=target-cur
        if abs(err)<tol: break
        a=[0,0,0,0,vac]; a[idx]=float(np.clip(err,-lim,lim))
        prev=cur
        obs,*_=step(env,*a)
        r2=d(obs,'robot')
        if abs(r2[field]-prev)<1e-12 and abs(err)>tol:
            return obs,False
    return obs,True
def setx(env,obs,t,vac=0): return move_to(env,obs,'x',t,0,0.05,vac)
def sety(env,obs,t,vac=0): return move_to(env,obs,'y',t,1,0.05,vac)
def setarm(env,obs,t,vac=0): return move_to(env,obs,'arm_joint',t,3,0.1,vac)
