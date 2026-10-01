import numpy as np
from types import SimpleNamespace
from packing import plan_parts
class State:
 def get(self,o,f):
  if o.name=='rack':return {'half_extent_x':.1,'half_extent_y':.15}[f]
  return {'half_extent_x':.05,'half_extent_y':.05,'side_a':.1,'side_b':.1,'triangle_type':o.kind}[f]
s=State();rack=SimpleNamespace(name='rack')
for count in range(3,7):
 for cubes in range(count+1):
  parts=[SimpleNamespace(name='part'+str(i),kind=1,type=SimpleNamespace(name='Kinematic3DCuboid' if i<cubes else 'Kinematic3DTriangle')) for i in range(count)]
  result=plan_parts(s,parts,rack)
  print('N',count,'C',cubes,'fits',len(result),{k:(v[0].round(3).tolist(),round(v[1],2)) for k,v in result.items()},flush=True)
