import sys, numpy as np, approach
from common import make_env
from approach import GeneratedApproach
from geom import shapes
p=tuple(float(v) for v in sys.argv[1].split(",")); sd=int(sys.argv[2]); c=int(sys.argv[3])
approach.FAR_PARAMS=[p]
env=make_env(); s,info=env.reset(seed=sd,options={'object_count':c})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
for t in range(400):
    a=ap.get_action(s); s,*_=env.step(a)
    ap._parse(s)
    if ap.phase=='grasp' and ap.far_mode and (abs(a[0])<0.007 and abs(a[1])<0.007) :
        d=shapes(env); lf=np.array(d['left_finger']); hk=np.array(d['hook'][0])
        r=ap.robot; th=r['theta']; gx=r['x']+r['arm_joint']*np.cos(th); gy=r['y']+r['arm_joint']*np.sin(th)
        def rel(pt):
            dx,dy=pt[0]-gx,pt[1]-gy; return (round(np.cos(th)*dx+np.sin(th)*dy,4), round(-np.sin(th)*dx+np.cos(th)*dy,4))
        print(t,ap.wait,ap.grasp_tries,np.round(a,4),'rx',round(r['x'],4),'gap',round(r['finger_gap'],4),'hookth',round(ap.hook['theta']+np.pi/2,4),'held',ap.hook['held'],
              'fingerRight', np.round(lf[:,0].max(),4), 'hookLeft', np.round(hk[:,0].min(),4), 'P rel', rel((hk[:,0].min(), hk[:,1].max())), 'corner rel', rel((ap.hook['x'],ap.hook['y'])))
    if ap.grasp_tries>0 or ap.hook['held']>0.5: break
