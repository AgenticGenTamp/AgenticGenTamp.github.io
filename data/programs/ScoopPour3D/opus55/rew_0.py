import numpy as np
from calib_util import *
r=R(0)
names=r.obs.get_object_names(); print(names)
for n in names:
    o=r.obs.get_object_from_name(n)
    print(n, o.type if hasattr(o,'type') else '', [ (f, round(float(r.obs.get(o,f)),4)) for f in ['x','y','z']] if n!='robot' else '')
print('base',r.base(),'fk',r.fk(tool=-0.224)[0] if False else '')
