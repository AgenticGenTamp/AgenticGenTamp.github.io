import sys, numpy as np
from simutil import make_env
from approach import GeneratedApproach
import imageio
N=int(sys.argv[1]); sd=int(sys.argv[2]); steps=[int(x) for x in sys.argv[3].split(',')]
env=make_env(N); ap=GeneratedApproach(env.action_space,None,{})
s,info=env.reset(seed=sd,options={'object_count':N}); ap.reset(s,info)
inner=env
for k in range(max(steps)+1):
    a=ap.get_action(s); s,r,term,tr,_=env.step(a)
    if k in steps:
        try:
            img=env.render()
        except Exception as e:
            print('render err',e); img=None
        if img is None:
            e=env
            while hasattr(e,'env') or hasattr(e,'_object_centric_env'):
                e=getattr(e,'_object_centric_env',None) or e.env
                try: img=e.render(); 
                except Exception: pass
                if img is not None: break
        if img is not None: imageio.imwrite('snap_%d_%d.png'%(sd,k),img); print('saved',k,img.shape)
    if term: break
