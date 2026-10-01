from probe_push_lib import *
import sys
for s in [135,44,1]:
    p=P(s); print(s,p.h()['x'],std_grasp(p),p.h()); p.env.close()
