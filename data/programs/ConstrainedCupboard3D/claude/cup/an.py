import numpy as np
def load(p):
    f=open(p,'rb'); assert f.readline().strip()==b'P6'
    l=f.readline()
    while l.startswith(b'#'): l=f.readline()
    w,h=map(int,l.split()); f.readline()
    return np.frombuffer(f.read(),dtype=np.uint8).reshape(h,w,3)
for s in [0,1,7]:
    a=load('cup/s%d.ppm'%s)
    cols,cnt=np.unique(a.reshape(-1,3),axis=0,return_counts=True)
    o=np.argsort(-cnt)[:14]
    print('seed',s, a.shape)
    for i in o: print('  ',cols[i],cnt[i])
