from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
rob = obs.get_object_from_name('robot')
movs = obs.get_objects([t for t in env.observation_space.types if t.name=='mujoco_movable_object'][0])
def show(t,rew):
    s=f"{t} r={rew:.3f} base=({obs.get(rob,'pos_base_x'):.2f},{obs.get(rob,'pos_base_y'):.2f},{obs.get(rob,'pos_base_rot'):.2f}) "
    for m in movs: s+=f"{m.name}=({obs.get(m,'x'):.2f},{obs.get(m,'y'):.2f},{obs.get(m,'z'):.2f}) "
    print(s)
for t in range(40):
    a = np.zeros(11, dtype=np.float32)
    a[1]=-0.05
    obs, rew, term, trunc, info = env.step(a)
    if t%3==0: show(t,rew)
env.close()
