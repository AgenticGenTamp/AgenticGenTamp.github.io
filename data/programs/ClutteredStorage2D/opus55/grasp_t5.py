from grasp_utils import *
env = new_env()
B='block3'
def moved(obs0,obs1): return np.abs(blk(obs1,B)-blk(obs0,B)).max()>1e-6
for gap in [0.02, 0.005, 0.001, 0.0]:
    obs, info = env.reset(seed=0)
    obs,ok,n,_ = drive_to(env,obs,2.97,1.35,np.pi/2,0.2)
    obs,ok,n,_ = drive_to(env,obs,3.84,1.35,np.pi/2,0.2)
    obs,n = creep(env,obs,dy=0.01); yc=rob(obs)[1]
    obs,ok,n,_ = drive_to(env,obs,3.84,yc-gap,np.pi/2,0.2)
    obs,*_=env.step(act(vac=1)); o0=obs
    obs,*_=env.step(act(dy=-0.05,vac=1)); print('below gap',gap,'grasped',moved(o0,obs))
# lateral offset: gripper center x offset relative block span 3.70-3.98 (gripper half-height .07)
for xo in [3.64, 3.68, 3.72, 3.95, 4.0, 4.04]:
    obs, info = env.reset(seed=0)
    obs,ok,n,_ = drive_to(env,obs,2.97,1.35,np.pi/2,0.2)
    obs,ok,n,_ = drive_to(env,obs,xo,1.35,np.pi/2,0.2)
    obs,n = creep(env,obs,dy=0.01); yc=rob(obs)[1]
    obs,*_=env.step(act(vac=1)); o0=obs
    obs,*_=env.step(act(dy=-0.05,vac=1)); print('xoff',xo,'contact tip',round(yc+.21,4),'grasped',moved(o0,obs))
# short end from right
obs, info = env.reset(seed=0)
obs,ok,n,_ = drive_to(env,obs,2.97,1.35,np.pi/2,0.2)
obs,ok,n,_ = drive_to(env,obs,4.4,1.35,np.pi,0.2,log=True)
obs,ok,n,_ = drive_to(env,obs,4.4,1.616,np.pi,0.2,log=True)
obs,n = creep(env,obs,dx=-0.01); r=rob(obs); print('end contact x',r[0],'tip',r[0]-.21)
obs,*_=env.step(act(vac=1)); o0=obs
obs,*_=env.step(act(dx=0.05,vac=1)); print('end grasp', moved(o0,obs))
obs,*_=env.step(act(dth=0.1,vac=1)); print('end grasp rot', blk(obs,B)-blk(o0,B))
