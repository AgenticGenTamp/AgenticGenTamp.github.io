from test_batch import *
from approach import GeneratedApproach as Base
class WallAim(Base):
 def xyz(self,state,obj):
  p=super().xyz(state,obj)
  if obj.name.startswith('bin_'):p[0]+=.09
  return p
# Use the same test harness with a temporary policy binding.
import test_batch
test_batch.GeneratedApproach=WallAim
with ThreadPoolExecutor(max_workers=3) as pool:
 for r in pool.map(test_batch.run,[(9,2),(5,2),(21,2),(0,2),(13,2),(11,1)]):print(json.dumps(r),flush=True)
