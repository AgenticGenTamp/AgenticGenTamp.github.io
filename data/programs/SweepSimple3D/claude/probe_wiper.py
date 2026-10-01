from env_client import make_env
import numpy as np
env=make_env()
obs,info=env.reset(seed=1)
def g(n,f): return float(obs.get(obs.get_object_from_name(n),f))
print("wiper",[round(g("wiper_0",f),3) for f in ["x","y","z","qw","qx","qy","qz"]])
print("cube",round(g("cube_0","x"),3),round(g("cube_0","y"),3))
wx=g("wiper_0","x")
for i in range(120):
    a=np.zeros(11,dtype=np.float32)
    bx=g("robot","pos_base_x")
    a[0]=np.clip(wx-bx,-0.1,0.1)
    if abs(wx-bx)<0.03: a[1]=-0.05
    obs,rew,term,trunc,info=env.step(a)
    if i%10==0:
        print(i,round(rew,3),"rob",round(g("robot","pos_base_x"),3),round(g("robot","pos_base_y"),3),
              "wiper",[round(g("wiper_0",f),3) for f in ["x","y","z","qw","qz"]],
              "cube",round(g("cube_0","x"),3),round(g("cube_0","y"),3))
    if g("robot","pos_base_y")<0.2: break
env.close()
