import numpy as np, sys
from probe_vis_lib import *
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 39
env=make_env(); obs,info=env.reset(seed=SEED, options={'object_count':1})
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
print("seed",SEED,"O",np.round(O,4))
free,xs,res=build_grid(obs)
# verify image requires visibility
obs,_=nav(env,obs,1.0,1.0,i=0,tol=0.02,free=free,xs=xs,res=res)
obs,v=vis_test(env,obs,0); print("ref pos (1,1) visible:",v)
obs,_,_,_,_=st(env,op='calibrate',i=0); print("calibrated:",rf(obs,0)['calibrated'])
obs,_=nav(env,obs,2.0,0.2,i=0,tol=0.02,free=free,xs=xs,res=res)
obs,_,_,_,_=st(env,op='image',i=0)
print("after image at far pos: calib=%.0f have=%.0f (image refused if calib stays 1)"%(rf(obs,0)['calibrated'],feats(obs,'objective0')['have_image_rover0']))
obs,_=nav(env,obs,1.0,1.0,i=0,tol=0.02,free=free,xs=xs,res=res)
obs,_,_,_,_=st(env,op='image',i=0); print("back at ref, image -> calib=%.0f have=%.0f"%(rf(obs,0)['calibrated'],feats(obs,'objective0')['have_image_rover0']))
ygrid=np.arange(0.2,2.01,0.2); xgrid=np.arange(0.4,2.01,0.2)
recs=[]
for r,y in enumerate(ygrid[::-1]):
    cells={}
    cols = xgrid if r%2==0 else xgrid[::-1]
    for x in cols:
        obs,ok=goto(env,obs,x,y,i=0,tol=0.02)
        p=pose(obs,0)[:2]
        if np.linalg.norm(p-np.array([x,y]))>0.05:
            obs,ok=nav(env,obs,x,y,i=0,tol=0.02,free=free,xs=xs,res=res); p=pose(obs,0)[:2]
        if np.linalg.norm(p-np.array([x,y]))>0.05: cells[x]='?'; continue
        obs,v=vis_test(env,obs,0)
        cells[x]='#' if v else '.'
        recs.append((float(p[0]),float(p[1]),int(v)))
    print("y=%+.1f  %s"%(y,"".join(cells[x] for x in xgrid)))
print("x:       "+"".join("%d"%int(round(x*10))%10 if False else "%s"%str(int(round(x*10)))[-1] for x in xgrid))
np.save('vis_grid_%d.npy'%SEED,np.array(recs))
env.close()
