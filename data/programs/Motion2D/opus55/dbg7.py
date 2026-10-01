import sys, math
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
env = make_env(); seed=int(sys.argv[1])
obs, info = env.reset(seed=seed, options={'object_count':8})
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap._parse(obs)
ap.arm=0.1
from scipy import ndimage
for th in [math.pi/2, -math.pi/2, 0.0]:
    for m in [0.0005,-0.0001]:
        ap.ptheta=th; ap.margin=m
        e=m+2e-5; r=0.1+e
        xs=np.arange(0.005,2.5,0.01); 
        crit=[]
        for (x,y,t,w,h) in ap.obstacles: crit += [x-r, x+w+r]
        xs=np.sort(np.concatenate([xs,crit])); ys=np.arange(0.0025,2.5,0.005)
        GX,GY=np.meshgrid(xs,ys,indexing='ij'); F=ap._free(GX,GY)
        lab,n=ndimage.label(F, structure=np.ones((3,3)))
        rx,ry=float(obs.get(ap.robot,'x')),float(obs.get(ap.robot,'y'))
        i=np.argmin(abs(xs-rx)); j=np.argmin(abs(ys-ry)); L=lab[i,j]
        reach=GX[lab==L]
        print(th, m, 'max x reachable', reach.max() if L else None, 'free cols in corridor', [round(x,4) for x in xs[(xs>0.55)&(xs<0.68)] if F[list(xs).index(x)].any()])
