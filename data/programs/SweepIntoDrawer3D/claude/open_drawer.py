"""Open a kitchen-island drawer. Usage: python open_drawer.py [seed] [wy] [wz]"""
import sys, numpy as np
from env_client import make_env
from ctrl2 import move_base, RFWD
from probe_drawer_lib import moveto_w, hold, tip_world

def rollx(R,a):
    c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])@R
RROLL=rollx(RFWD,np.pi/2)   # fingers separate horizontally

def attempt(env,obs,wy,wz,R=RROLL,bx=1.70,pushx=0.90,verbose=True):
    rep={}
    obs=move_base(env,obs,[bx,obs[126],obs[127]],steps=55,grip=1.0)
    obs,d1=moveto_w(env,obs,(1.20,wy,wz),R=R,steps=300,grip=1.0,multi=True,stall_n=40)
    rep['w1']=(d1['tip'].tolist(),round(d1['err'],3))
    obs,d2=moveto_w(env,obs,(pushx,wy,wz),R=R,steps=220,grip=1.0,multi=False,stall_n=30)
    rep['w2']=(d2['tip'].tolist(),round(d2['err'],3))
    q=obs[128:135].copy()
    obs=hold(env,obs,q,12,grip=0.0)           # close on handle
    dr0=obs[103:109].copy()
    # pull: drive base +x holding arm joints
    for i in range(160):
        a=np.zeros(11); a[0]=0.1; a[3:10]=np.clip(q-obs[128:135],-0.1,0.1); a[10]=0.0
        obs,r,t,tr,_=env.step(a)
        if obs[125]>1.97: break
    rep['drawer']=list(np.round(obs[103:109],3))
    rep['dmax']=float(np.max(np.abs(obs[103:109])))
    rep['base']=round(float(obs[125]),3)
    return obs,rep

if __name__=="__main__":
    seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
    wy=float(sys.argv[2]) if len(sys.argv)>2 else -0.10
    wz=float(sys.argv[3]) if len(sys.argv)>3 else 0.30
    env=make_env(); obs,_=env.reset(seed=seed)
    obs,rep=attempt(env,obs,wy,wz)
    print(seed,wy,wz,rep); env.close()
