from grasp_utils import *
env = new_env()
B='block3'
def moved(obs0,obs1): return np.abs(blk(obs1,B)-blk(obs0,B)).max()>1e-6
def trial(xo, gap):
    obs, info = env.reset(seed=0)
    obs,ok,n,_ = drive_to(env,obs,2.97,1.35,np.pi/2,0.2)
    obs,ok,n,_ = drive_to(env,obs,3.84,1.35,np.pi/2,0.2)
    obs,n = creep(env,obs,dy=0.01); yc=rob(obs)[1]
    obs,ok,n,_ = drive_to(env,obs,3.84,yc-gap,np.pi/2,0.2)
    obs,ok,n,_ = drive_to(env,obs,xo,yc-gap,np.pi/2,0.2)
    obs,*_=env.step(act(vac=1)); o0=obs
    obs,*_=env.step(act(dy=-0.05,vac=1)); return moved(o0,obs)
#for gap in [0.03,0.04,0.05,0.08]: print('gap',gap,trial(3.84,gap))
#for xo in [3.55,3.6,3.62,4.06,4.1]: print('xo',xo,trial(xo,0.0))
print('---')
for gap in [0.022,0.025,0.028]: print('gap',gap,trial(3.84,gap))
for xo in [3.55,3.6,3.62,3.64]: print('xo gap.012',xo,trial(xo,0.012))
