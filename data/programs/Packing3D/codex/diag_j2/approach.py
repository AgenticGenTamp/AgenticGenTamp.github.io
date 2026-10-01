import numpy as np

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives): self.i=0
    def reset(self,state,info): self.i=0
    def get_action(self,s):
        a=np.zeros(11,np.float32)
        def g(n,f): return s.get(s.get_object_from_name(n),f)
        if self.i < 2:
            a[0]=np.clip(g('part0','pose_x')-g('rack','pose_x')-(g('robot','pos_base_x')+.12),-.2,.2)
            a[1]=np.clip(g('part0','pose_y')-g('rack','pose_y')-g('robot','pos_base_y'),-.2,.2)
            a[10]=1
        elif self.i < 12:
            a[4]=.1; a[10]=-1
        else: a[10]=-1
        self.i+=1
        return a
