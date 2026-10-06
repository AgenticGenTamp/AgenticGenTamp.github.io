import numpy as np
exec(open('approach.py').read())
s=Sim(1); st=s.get_state(); P=Planner(s)
q0=np.array(st[:10])
arm=np.array([0.85,-0.22,1.37,-1.85,1.09,-1.76,2.05])
for th in [-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8]:
    row=[]
    for y in [-1.0,-0.8,-0.6,-0.4,0,0.4,0.6,0.8,1.0]:
        for x in np.arange(-0.9,-0.3,0.01):
            q=np.concatenate([[x,y,th],arm])
            if P.table_collision(q): row.append(round(x-0.01,2)); break
        else: row.append(None)
    print(th,row)
