import sys, os, numpy as np
import approach as ap
from env_client import make_env
cls=[v for k,v in vars(ap).items() if isinstance(v,type) and hasattr(v,'_grasp_pose')][0]
SLOW=float(os.environ.get('SLOW',0))  # max |dy| when within 0.12 of gy
if SLOW:
    orig=cls._toward
    def _toward(self,r,x=None,y=None,th=None,arm=None,gap=None):
        a,d=orig(self,r,x,y,th,arm,gap)
        if self.phase=='descend_hook' and y is not None and r['y']<y+0.12: a[1]=max(a[1],-SLOW)
        return a,d
    cls._toward=_toward
if os.environ.get('LATCH'):
    ogp=cls._grasp_pose
    def _gp(self,h):
        if self.phase in ('descend_hook','close'):
            if getattr(self,'_gp_cache',None) is None: self._gp_cache=ogp(self,h)
            return self._gp_cache
        self._gp_cache=None; return ogp(self,h)
    cls._grasp_pose=_gp
env=make_env(); obs,info=env.reset(seed=int(sys.argv[1])); pol=cls(None,None,None); pol.reset(obs,info)
log=[]
for t in range(400):
    a=pol.get_action(obs); ph=pol.phase
    obs,*_=env.step(a)
    h=pol._hook(obs); r=pol._robot(obs)
    if ph in('descend_hook','close','regrasp'): log.append((t,ph[:3],round(r["x"],4),round(r["y"],4),round(h['x'],4),round(h['th'],3),h['held']))
    if h['held']>0.5 and ph=='close': print('HELD at step',t,'r',r,'h',h); break
else: print('no hold')
for l in log[:60]: print(l)
