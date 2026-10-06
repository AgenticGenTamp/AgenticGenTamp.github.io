import pickle, glob, time, numpy as np
from approach import *
from reach import *
tot=0; fails=0; prun=0; wrong=0; tt=0
for fn in sorted(glob.glob('logs/q_*.pkl')):
    for chk,start,goals,tb,ok,dt in pickle.load(open(fn,'rb')):
        t=time.time(); lab,res=base_reach_labels(chk.obs, chk.robot.r); tt+=time.time()-t
        ls=cell(lab,res,start)
        reach = ls>0 and any(cell(lab,res,g)==ls for g in goals)
        tot+=1; fails+= (not ok)
        if not reach: prun+=1; wrong+= ok
print('tot',tot,'fails',fails,'pruned',prun,'wrong(pruned but solved)',wrong,'time/q',tt/tot)
