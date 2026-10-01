import numpy as np
from env_client import make_env
from pngtool import readpng
M=np.load('cam_M.npy')
def pr(P):
    h=M@np.r_[np.asarray(P,float),1.0]; return np.array([h[0]/h[2],h[1]/h[2]])
env=make_env(); obs,_=env.reset(seed=0)
st=obs.copy(); st[125]=3.0; st[126]=-2.0; st[147:163]=0.0
img=readpng(env.render_state(state=st.tolist(),label="face2")).astype(int)
def px(P):
    u,v=pr(P); u=int(round(u)); v=int(round(v))
    return img[v,u] if (0<=v<480 and 0<=u<640) else np.array([-1,-1,-1])
print('fine y edges at z=0.25 x=0.90:')
for y in list(np.arange(-1.15,-1.04,0.01))+list(np.arange(0.93,1.02,0.01)):
    print('  y=%.3f %s'%(y,px([0.90,y,0.25])))
print('fine z top edge (x=0.90):')
for y in (-0.6,0.0,0.6):
  for z in np.arange(0.462,0.428,-0.002):
    print('  y=%.1f z=%.3f %s'%(y,z,px([0.90,y,z])))
print('fine z bottom (x=0.90, y=0.0):')
for z in np.arange(0.16,0.02,-0.01):
    print('  z=%.3f %s'%(z,px([0.90,0.0,z])))
env.close()
