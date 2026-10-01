from env_client import make_env
import numpy as np

def get(s, name, fs):
    o=s.get_object_from_name(name)
    return [round(float(s.get(o,f)),3) for f in fs]

tests = {
 "dx": [0.05,0,0,0,0], "dy":[0,0.05,0,0,0],
 "rot":[0,0,0.196,0,0], "arm":[0,0,0,0.1,0], "close":[0,0,0,0,-0.02]
}
fs=['x','y','theta','arm_length','finger_gap','vx_base','vy_base','omega_base','vx_arm','vy_arm']
for k,a in tests.items():
 e=make_env(); s,_=e.reset(seed=0)
 print(k,'before',get(s,'robot',fs))
 for i in range(3):
  s,r,t,tr,info=e.step(np.asarray(a, dtype=np.float64) * .99)
  print(i,get(s,'robot',fs),'block',get(s,'target_block',['x','y','theta','held']))
 e.close()
