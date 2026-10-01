import numpy as np
from env_client import make_env

def main():
    e=make_env(); o,info=e.reset(seed=0)
    print('reset info',info,'obj',o[:3],o[16:19],o[32:35],'robot',o[93:104],flush=True)
    print('kitchen',o[48:55],o[64:71],flush=True)
    for j in range(11):
        o,_=e.reset(seed=0); a=np.zeros(11,dtype=np.float32);a[j]=0.1 if j<10 else 1
        p,r,t,tr,i=e.step(a)
        print('j',j,'dq',np.round(p[93:104]-o[93:104],5),'reward',r,'info',i,'objects_delta',np.round(np.r_[p[:3]-o[:3],p[16:19]-o[16:19],p[32:35]-o[32:35]],5),flush=True)
    e.close()
if __name__=='__main__':main()
