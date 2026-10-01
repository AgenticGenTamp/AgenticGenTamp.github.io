import numpy as np, pickle
import robot
for s in [0,1,2]:
    d=pickle.load(open(f'calib_{s}.pkl','rb'))
    G=d['G']; errs=[]
    for q,T,b in d['data']:
        P = robot.base_tf(*b) @ robot.fk_tool(q) @ G
        errs.append(np.linalg.norm(P[:3,3]-T[:3,3]))
    print(s, np.mean(errs), np.max(errs))
