from probe_lib import *
pr=P(0)
for t in range(200):
    pr.step(np.zeros(11))
    if t%40==0: print(t,pr.obs[16:23],pr.obs[32:39][:3])
