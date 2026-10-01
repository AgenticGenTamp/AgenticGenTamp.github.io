import numpy as np
from probe_vis_lib import *
env=make_env(); obs,info=env.reset(seed=39, options={'object_count':1})
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
free,xs,res=build_grid(obs)
obs,ok0=nav(env,obs,1.4,1.2,i=0,tol=0.02,free=free,xs=xs,res=res)
obs,ok1=nav(env,obs,-0.4,1.2,i=1,tol=0.02,free=free,xs=xs,res=res)
print("poses",np.round(pose(obs,0)[:2],3),np.round(pose(obs,1)[:2],3),ok0,ok1)
def show(tag): print(tag,"c0=%.0f c1=%.0f have0=%.0f have1=%.0f"%(rf(obs,0)['calibrated'],rf(obs,1)['calibrated'],feats(obs,'objective0')['have_image_rover0'],feats(obs,'objective0')['have_image_rover1']))
obs,_,_,_,_=st(env,op='calibrate',i=0); show("r0 calib:")
obs,_,_,_,_=st(env,op='image',i=0);     show("r0 image:")
obs,_,_,_,_=st(env,op='calibrate',i=1); show("r1 calib:")
obs,_,_,_,_=st(env,op='calibrate',i=0); show("r0 calib while r1 calibrated:")
obs,_,_,_,_=st(env,op='image',i=1);     show("r1 image:")
obs,_,_,_,_=st(env,op='calibrate',i=0); show("r0 calib after r1 cleared:")
# simultaneous both calibrate in same step
obs,_,_,_,_=st(env,op='image',i=0); show("r0 image:")
a=np.zeros(8,dtype=np.float32); a[3]=OPS['calibrate']; a[7]=OPS['calibrate']
obs,_,_,_,_=env.step(a); show("both calibrate same step:")
env.close()
