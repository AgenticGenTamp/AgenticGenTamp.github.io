import numpy as np, kutil
from env_client import make_env
env=make_env(); o,_=env.reset(seed=1)
def setpose(st,name,p):
    ob=st.get_object_from_name(name)
    for f,v in zip(["pose_x","pose_y","pose_z"],p): st.set(ob,f,float(v))
def trial(bz,cz,bxy=(0.6,0.0),cxy=(0.6,-0.3)):
    o2,_=env.reset(seed=1)
    st=env.get_state()
    setpose(st,"box0",[bxy[0],bxy[1],bz]); setpose(st,"cube0",[cxy[0],cxy[1],cz])
    env.set_state(st)
    obs,r,t,tr,info=env.step(np.zeros(11,dtype=np.float32))
    return t,r
print(type(env.get_state()))
for bz,cz in [(0.5,0.425),(0.5010,0.4253),(0.5,0.4253),(0.5010,0.425),(0.502,0.425),(0.51,0.425),(0.5,0.43),(0.5,0.5)]:
    print(bz,cz,trial(bz,cz))
env.close()
