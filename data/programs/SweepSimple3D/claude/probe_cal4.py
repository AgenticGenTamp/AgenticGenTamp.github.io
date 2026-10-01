import numpy as np, fk, ctrl
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=6)
def g(n,f): return float(obs.get(obs.get_object_from_name(n),f))
def J(): return np.array([g("robot","pos_arm_joint%d"%i) for i in range(1,8)])
def B(): return np.array([g("robot","pos_base_x"),g("robot","pos_base_y"),g("robot","pos_base_rot")])
def C(): return np.array([g("cube_0","x"),g("cube_0","y")])
def step(tq,tb=None,grip=0.0):
    global obs
    a=ctrl.action(J(),tq,B(),tb,grip); obs,rew,t,tr,i2=env.step(a)
def tip(): 
    p,_=fk.fk(J(),*B()); return p

def trial(L, d):
    d=np.array(d,dtype=float); d/=np.linalg.norm(d)
    q=ctrl.ik_local(L,J())
    c=C(); b=B()
    # predicted tip offset from base (world), for current config, at base yaw b[2]
    # solve base pos so that pred tip = c - 0.45*d  (in xy)
    for _ in range(90): step(q,None)  # get arm to config first
    p=tip(); off=p[:2]-B()[:2]
    tb=np.array([ (c-0.45*d - off)[0], (c-0.45*d-off)[1], b[2] ])
    for _ in range(80): step(q,tb)
    c=C()
    for i in range(120):
        tb2=np.array([B()[0]+d[0]*0.03, B()[1]+d[1]*0.03, b[2]])
        step(q,tb2)
        nc=C()
        if np.linalg.norm(nc-c)>0.004:
            p=tip(); print("L",L,"d",np.round(d,2),"contact: predtip",np.round(p,3),"cube",np.round(c,3),"resid(cube-tip)",np.round(c-p[:2],3))
            return
    print("L",L,"d",np.round(d,2),"NO CONTACT, tip",np.round(tip(),3),"cube",np.round(C(),3))

trial([0.45,0.0,-0.02],[0,-1])
trial([0.65,0.0,-0.02],[0,-1])
trial([0.45,0.0,-0.02],[1,0])
trial([0.45,0.0,-0.02],[0,1])
env.close()
