from grip_util import *
from multiprocessing import Pool
def run(seed):
    g=G(seed); c=g.cubes(); b=g.bins()['bin_yellow_0']
    iso=[(n,p.round(3).tolist(),round(float(np.degrees(gyaw(g,n))),1)) for n,p in c.items() if isolated(c,n,0.04) and abs(p[0]-b[0])<b[3]/2-0.03 and abs(p[1]-b[1])<b[4]/2-0.03 and abs(p[2]-0.4825)<0.002]
    g.env.close(); return seed, iso
with Pool(4) as p:
    for s,iso in p.imap_unordered(run,range(30)): print(s,iso,flush=True)
