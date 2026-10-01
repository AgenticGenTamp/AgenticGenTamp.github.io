from probe_dyn_lib import *
from probe_dyn_b import exp
S=None
for v in [0.01,0.0499]:
    exp('c bar end -x push (from +x)',[0.8,-0.06],[-1,0],v,int(0.5/v))
    exp('c bar end +x push (from -x)',[-0.8,-0.06],[1,0],v,int(0.5/v))
    exp('c bar end -x at y=-0.01',[0.8,-0.01],[-1,0],v,int(0.5/v))
    exp('d stem side y-.6 push -x',[0.6,-0.6],[-1,0],v,int(0.6/v))
    exp('d stem side y-1.0 push -x',[0.6,-1.0],[-1,0],v,int(0.6/v))
    exp('d stem side y-.3 push +x',[-0.6,-0.3],[1,0],v,int(0.6/v))
    exp('bar under x.4 push up',[0.4,-0.5],[0,1],v,int(0.5/v))
