"""Offline interface checks using observations captured from live experiments."""
import ast
import json
from pathlib import Path
import numpy as np
from approach import GeneratedApproach

for filename in ['approach.py', 'kinematics.py']:
 tree=ast.parse(Path(filename).read_text())
 for node in ast.walk(tree):
  if isinstance(node,ast.Import):
   assert all(alias.name=='numpy' for alias in node.names)
  if isinstance(node,ast.ImportFrom):
   assert node.module=='kinematics'
class Space:
 low=np.array([-.1]*10+[0.],dtype=np.float32)
 high=np.array([.1]*10+[1.],dtype=np.float32)
p=GeneratedApproach(Space(),None,{})
count=0
for path in Path('.').glob('*state.json'):
 state=np.asarray(json.loads(path.read_text()))
 if state.shape!=(115,):continue
 p.reset(state,{})
 for _ in range(1000):
  a=p.get_action(state)
  assert a.shape==(11,) and a.dtype==np.float32
  assert np.all(np.isfinite(a))
  assert np.all(a>=Space.low) and np.all(a<=Space.high)
 count+=1
print('Import and action contract checks passed on',count,'saved observations.')
