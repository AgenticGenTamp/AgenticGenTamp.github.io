from env_client import make_env
import numpy as np
env = make_env()
tf = {t.name: list(f) for t,f in env.observation_space.type_features.items()}
obs,info = env.reset(seed=42)
r=obs.get_object_from_name("robot")
def show(tag):
    print(tag, {f: round(float(obs.get(r,f)),3) for f in tf['kin_robot'] if f in ('x','y','theta','arm_joint','arm_length','finger_gap')})
show("init")
for _ in range(5): obs,_,_,_,_=env.step(np.array([0,0,0,0.0999,0],dtype=np.float32))
show("arm+5")
for _ in range(20): obs,_,_,_,_=env.step(np.array([0,0,0,-0.0999,0],dtype=np.float32))
show("arm-20")
env.close()
