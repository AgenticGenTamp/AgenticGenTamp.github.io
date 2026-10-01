from grasp_utils import *
env = new_env()
B='block3'
obs, info = env.reset(seed=0)
obs,ok,n,_ = drive_to(env,obs,2.97,1.35,np.pi/2,0.2)
obs,ok,n,_ = drive_to(env,obs,3.84,1.35,np.pi/2,0.2,vac=1)
obs,n = creep(env,obs,dy=0.01,vac=1); b0=blk(obs,B)
obs,*_=env.step(act(dy=-0.05,vac=1)); print('vac-on-before-contact then retreat: block moved', np.abs(blk(obs,B)-b0).max())
obs,*_=env.step(act(dy=0.03,vac=1)); obs,*_=env.step(act(vac=1)); b0=blk(obs,B)
obs,*_=env.step(act(dy=-0.05,vac=1)); print('approach within range with vac on: moved', np.abs(blk(obs,B)-b0).max())
