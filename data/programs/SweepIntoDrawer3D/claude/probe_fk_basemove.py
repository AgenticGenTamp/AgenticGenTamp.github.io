import numpy as np
from env_client import make_env
from probe_fk_ctl import servo, grip_hold, ee_world
np.set_printoptions(precision=4,suppress=True,linewidth=250)
env=make_env(); o,_=env.reset(seed=0); o=np.asarray(o,float)
C0=o[:80].reshape(5,16)[:,:3].copy()
# move base: +y 0.15 and yaw -0.30
for i in range(40):
    a=np.zeros(11); a[1]=0.06; a[10]=0.0; o,_,_,_,_=env.step(a); o=np.asarray(o,float)
    if o[126]>-0.083+0.15: break
for i in range(40):
    a=np.zeros(11); a[2]=-0.05; a[10]=0.0; o,_,_,_,_=env.step(a); o=np.asarray(o,float)
    if o[127]<3.0922-0.30: break
for i in range(8):
    a=np.zeros(11); a[10]=0.0; o,_,_,_,_=env.step(a); o=np.asarray(o,float)
print("new base",np.round(o[125:128],4))
t=C0[2]  # cube2 (0.7271,-0.0178)
print("cube2",np.round(t,4))
o,e,u=servo(env,o,[t[0],t[1],0.58],steps=120,grip=0.0); print("above err",round(e,4),"ee",np.round(ee_world(o),4),"u",u)
o,e,u=servo(env,o,[t[0],t[1],0.4698],steps=100,grip=0.0); print("at err",round(e,4),"ee",np.round(ee_world(o),4),"cube2",np.round(o[:80].reshape(5,16)[2,:3],4))
o=grip_hold(env,o,1.0,15)
o,e,u=servo(env,o,[0.80,-0.25,0.63],steps=120,grip=1.0)
c=o[:80].reshape(5,16)[2,:3]
print("carry err",round(e,4),"ee",np.round(ee_world(o),4),"cube2",np.round(c,4),"resid",np.round(c-ee_world(o),4))
o,e,u=servo(env,o,[0.62,-0.35,0.55],steps=120,grip=1.0)
c=o[:80].reshape(5,16)[2,:3]
print("carry2 err",round(e,4),"ee",np.round(ee_world(o),4),"cube2",np.round(c,4),"resid",np.round(c-ee_world(o),4))
print("base",np.round(o[125:128],4))
env.close()
