import numpy as np, fk, ctrl
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=6)
def g(n,f): return float(obs.get(obs.get_object_from_name(n),f))
def J(): return np.array([g("robot","pos_arm_joint%d"%i) for i in range(1,8)])
def B(): return np.array([g("robot","pos_base_x"),g("robot","pos_base_y"),g("robot","pos_base_rot")])
def step(tq, tb=None, grip=0.0):
    global obs
    a=ctrl.action(J(),tq,B(),tb,grip); obs,rew,t,tr,i2=env.step(a); return rew,t
q=ctrl.ik_local([0.45,0.0,-0.02], J())
print("target q",np.round(q,3))
cx,cy=g("cube_0","x"),g("cube_0","y"); b0=B()
tb=np.array([cx+0.014,b0[1],b0[2]])
for i in range(90): step(q,tb)
print("base",np.round(B(),3),"jerr",round(float(np.max(np.abs(J()-q))),4))
p,_=fk.fk(J(),*B()); print("pred tip",np.round(p,3),"cube",round(cx,3),round(cy,3))
for i in range(220):
    step(q, np.array([tb[0], B()[1]-0.03, b0[2]]))
    ncx,ncy=g("cube_0","x"),g("cube_0","y")
    if abs(ncx-cx)>0.004 or abs(ncy-cy)>0.004:
        p,_=fk.fk(J(),*B()); print("CONTACT i",i,"base",np.round(B(),3),"pred tip",np.round(p,3),"cube",round(ncx,3),round(ncy,3)); break
    if B()[1]<cy-0.35: print("no contact"); break
for i in range(40): step(q, np.array([tb[0], B()[1]-0.03, b0[2]]))
p,_=fk.fk(J(),*B())
print("end base",np.round(B(),3),"pred tip",np.round(p,3),"cube",[round(g("cube_0",f),3) for f in "xyz"])
env.close()
