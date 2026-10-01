import batch_test
from approach import GeneratedApproach
from kinematics import grasp_joints
from concurrent.futures import ThreadPoolExecutor
class Higher(GeneratedApproach):
 def select_grasp(self,s,idx):
  super().select_grasp(s,idx)
  if idx==16 and s[idx+2]<.47:
   self.lowq=grasp_joints(.095)
   self.slide=False
batch_test.GeneratedApproach=Higher
with ThreadPoolExecutor(max_workers=2) as p:
 list(p.map(batch_test.run,[27,36]))
