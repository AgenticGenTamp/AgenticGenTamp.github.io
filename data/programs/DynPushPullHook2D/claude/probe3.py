from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=42)
def rb(o): 
    r=o.get_object_from_name("robot"); return (round(float(o.get(r,'x')),3), round(float(o.get(r,'y')),3), round(float(o.get(r,'theta')),3), round(float(o.get(r,'arm_length')),3), round(float(o.get(r,'finger_gap')),3))
print("start", rb(obs))
# move up
for i in range(100):
    obs,r,t,tr,info = env.step(np.array([0,0.0499,0,0,0],dtype=np.float32))
print("after up", rb(obs))
for i in range(100):
    obs,r,t,tr,info = env.step(np.array([0.0499,0,0,0,0],dtype=np.float32))
print("after right", rb(obs))
for i in range(100):
    obs,r,t,tr,info = env.step(np.array([0,0.0499,0,0,0],dtype=np.float32))
print("after up2", rb(obs))
for i in range(60):
    obs,r,t,tr,info = env.step(np.array([0,0,0,0.0999,0],dtype=np.float32))
print("after arm out", rb(obs))
for i in range(60):
    obs,r,t,tr,info = env.step(np.array([0,0,0,0,0.0199],dtype=np.float32))
print("after grip open", rb(obs))
for i in range(60):
    obs,r,t,tr,info = env.step(np.array([0,0,0,0,-0.0199],dtype=np.float32))
print("after grip close", rb(obs))
for i in range(100):
    obs,r,t,tr,info = env.step(np.array([0,0,0.0653,0,0],dtype=np.float32))
print("after rot", rb(obs))
env.close()
