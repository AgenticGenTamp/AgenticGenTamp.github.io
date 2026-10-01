from probe_hook_lib import P
import numpy as np
for s in [135,44,1]:
    p=P(s); print(s,p.r(),p.h()); p.env.close()
