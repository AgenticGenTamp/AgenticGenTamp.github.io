import numpy as np, json, sys
from env_client import make_env
import kinova_fk as K
from calib import R_down, goto_q, move_line

MOUNT=np.array([0.0,0.0,0.44])

def world_to_rel(p, base):
    bx,by,th=base
    v=np.array([p[0]-bx,p[1]-by])
    c,s=np.cos(-th),np.sin(-th)
    r=np.array([c*v[0]-s*v[1], s*v[0]+c*v[1]])
    return np.array([r[0]-MOUNT[0], r[1]-MOUNT[1], p[2]-MOUNT[2]])

def run(seed=0, gripclose=1.0, zoff=0.0):
    env=make_env(); obs,info=env.reset(seed=seed)
    o=np.asarray(obs); base=o[93:96]; can=o[32:35].copy()
    tgt=can.copy(); tgt[2]=can[2]+zoff
    rel=world_to_rel(tgt,base)
    print("can",np.round(can,3),"rel",np.round(rel,3))
    above=rel+np.array([0,0,0.18])
    obs=move_line(env,obs,above,above,n=1,steps_per=45,grip=0.0)
    print("after above fk",np.round(K.fk(np.asarray(obs)[96:103])[:3,3],3), "can",np.round(np.asarray(obs)[32:35],3))
    obs=move_line(env,obs,above,rel,n=4,steps_per=8,grip=0.0)
    print("after down fk",np.round(K.fk(np.asarray(obs)[96:103])[:3,3],3),"can",np.round(np.asarray(obs)[32:35],3))
    for i in range(6):
        a=np.zeros(11,dtype=np.float32); a[10]=gripclose; obs,r,te,tr,inf=env.step(a)
    obs=move_line(env,obs,rel,rel+np.array([0,0,0.2]),n=4,steps_per=10,grip=gripclose)
    o=np.asarray(obs)
    print("after lift fk",np.round(K.fk(o[96:103])[:3,3],3),"can",np.round(o[32:35],3),"grip",o[103])
    json.dump(o.tolist(),open("grasp_snap.json","w"))
    env.close()

if __name__=="__main__":
    run(gripclose=float(sys.argv[1]) if len(sys.argv)>1 else 1.0, zoff=float(sys.argv[2]) if len(sys.argv)>2 else 0.0)
