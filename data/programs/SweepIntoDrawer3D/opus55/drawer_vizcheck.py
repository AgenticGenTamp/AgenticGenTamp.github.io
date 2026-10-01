import sys; sys.path.insert(0,'drawer_viz')
from approach import GeneratedApproach
from drawer_util import *
r=R(); ap=GeneratedApproach(None,None,{}); ap.reset(r.obs,{})
for t in range(240):
    r.step(ap.get_action(r.obs))
    if t%25==24: print(t+1,'base',r.base,'tool',r.tool()[:3,3],'cube0',r.obs[0:3],'dr',r.obs[103:109])
