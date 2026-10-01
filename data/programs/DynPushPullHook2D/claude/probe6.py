from env_client import make_env
import numpy as np
env = make_env()
obs,info = env.reset(seed=42)
r=obs.get_object_from_name("robot")
def g(f): return float(obs.get(r,f))
def step(a):
    global obs
    obs,_,_,_,_=env.step(np.array(a,dtype=np.float32))
# rotate to pi/2
while abs(g('theta')-np.pi/2)>0.01:
    d=np.clip(np.pi/2-g('theta'),-0.065,0.065); step([0,0,d*0.99,0,0])
print("theta",g('theta'))
for _ in range(60): step([0,0.0499,0,0,0])
print("y",g('y'),"armj",g('arm_joint'))
for _ in range(10): step([0,0,0,0.0999,0])
print("after extend: y",g('y'),"armj",g('arm_joint'))
for _ in range(20): step([0,0.0499,0,0,0])
print("after up: y",g('y'),"armj",g('arm_joint'))
env.close()
