import numpy as np, fk
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=6)
def g(n,f): return float(obs.get(obs.get_object_from_name(n),f))
def J(): return np.array([g("robot","pos_arm_joint%d"%i) for i in range(1,8)])
def B(): return np.array([g("robot","pos_base_x"),g("robot","pos_base_y"),g("robot","pos_base_rot")])
def step(tgtq, base_t=None, grip=0.0):
    global obs
    a=np.zeros(11,dtype=np.float32)
    if base_t is not None:
        a[0:3]=np.clip(np.array(base_t)-B(),-0.1,0.1)
    a[3:10]=np.clip(tgtq-J(),-0.1,0.1)
    a[10]=grip
    obs,rew,t,tr,i2=env.step(a); return rew,t
Rd=np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)
q=fk.ik(np.array([0.45,0.0,-0.02]),Rd,J(),(0,0,0))
cx,cy=g("cube_0","x"),g("cube_0","y")
b0=B()
# stage 1: raise arm to config while moving base x to cube x (keep y)
tgt_base=np.array([cx+0.014, b0[1], b0[2]])
for i in range(80): step(q, tgt_base)
print("after pose: base",np.round(B(),3),"jerr",round(float(np.max(np.abs(J()-q))),3))
p,_=fk.fk(J(),B()[0],B()[1],B()[2]); print("pred tip world",np.round(p,3),"cube",round(cx,3),round(cy,3))
# stage 2: drive base -y slowly until cube moves
for i in range(200):
    tb=np.array([tgt_base[0], B()[1]-0.03, b0[2]])
    step(q, tb)
    ncx,ncy=g("cube_0","x"),g("cube_0","y")
    if abs(ncx-cx)>0.005 or abs(ncy-cy)>0.005:
        p,_=fk.fk(J(),B()[0],B()[1],B()[2])
        print("CONTACT at step",i,"base",np.round(B(),3),"pred tip",np.round(p,3),"cube",round(ncx,3),round(ncy,3))
        break
    if B()[1] < cy-0.3:
        print("no contact, base y",B()[1]); break
# continue pushing 30 steps and report relation
for i in range(40):
    tb=np.array([tgt_base[0], B()[1]-0.03, b0[2]]); step(q,tb)
p,_=fk.fk(J(),B()[0],B()[1],B()[2])
print("end: base",np.round(B(),3),"pred tip",np.round(p,3),"cube",round(g("cube_0","x"),3),round(g("cube_0","y"),3),round(g("cube_0","z"),3))
env.close()
