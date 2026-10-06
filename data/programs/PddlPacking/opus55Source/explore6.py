import numpy as np
exec(open('approach.py').read())
s=Sim(1); st=s.get_state(); q0=np.array(st[:10])
P=Planner(s)
arm=q0[3:].copy()
for th,name in [(0,'near'),(np.pi,'far'),(np.pi/2,'right'),(-np.pi/2,'left')]:
    d=np.array([np.cos(th),np.sin(th)])
    for dist in np.arange(1.2,0.3,-0.02):
        # base center placed at -d*dist from table center edge-normal
        c = -d*dist
        if name in('near','far'): c=c+np.array([0,0]) 
        q=np.concatenate([[c[0],c[1],th],arm])
        if P.table_collision(q):
            print(name,'collides at |pos|', round(dist,2)); break
