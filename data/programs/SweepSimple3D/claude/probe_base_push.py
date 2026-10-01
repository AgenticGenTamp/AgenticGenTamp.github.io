from env_client import make_env
import numpy as np
env=make_env(); T=env.observation_space.get_type
obs,info=env.reset(seed=1)
def g(n,f): return float(obs.get(obs.get_object_from_name(n),f))
print("cube",g("cube_0","x"),g("cube_0","y"),"rob",g("robot","pos_base_x"),g("robot","pos_base_y"))
tx=g("cube_0","x"); 
for i in range(200):
    a=np.zeros(11,dtype=np.float32)
    bx=g("robot","pos_base_x"); by=g("robot","pos_base_y")
    a[0]=np.clip(tx-bx,-0.1,0.1)
    if abs(tx-bx)<0.02: a[1]=-0.05
    obs,rew,term,trunc,info=env.step(a)
    if i%10==0:
        print(i,round(rew,3),"rob",round(g("robot","pos_base_x"),3),round(g("robot","pos_base_y"),3),
              "cube",round(g("cube_0","x"),3),round(g("cube_0","y"),3),round(g("cube_0","z"),3))
    if g("robot","pos_base_y")<0.1: break
env.close()
