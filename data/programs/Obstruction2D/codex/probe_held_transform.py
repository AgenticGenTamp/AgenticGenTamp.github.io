import numpy as np
from env_client import make_env

e=make_env(); s,_=e.reset(seed=0); T=e.observation_space
r=s.get_objects(T.get_type("crv_robot"))[0]; b=s.get_objects(T.get_type("target_block"))[0]
tx=float(s.get(b,"x"))+.05
for _ in range(20):
 dx=np.clip(tx-float(s.get(r,"x")),-.05,.05); s,*_=e.step(np.array([dx,0,0,0,1],np.float32))
 if abs(dx)<1e-6: break
s,*_=e.step(np.array([0,0,0,.1,1],np.float32))
for _ in range(30): s,*_=e.step(np.array([0,-.05,0,0,1],np.float32))
def p(tag): print(tag,"r",*[round(float(s.get(r,f)),6) for f in ("x","y","theta","arm_joint","vacuum")],"b",*[round(float(s.get(b,f)),6) for f in ("x","y","theta")])
p("contact")
for a in ([0,.05,0,0,1],[0,0,0,-.05,1],[0,0,.1963495,0,1],[.03,.02,0,.02,1]):
 s,*_=e.step(np.array(a,np.float32)); p(str(a))
e.close()
