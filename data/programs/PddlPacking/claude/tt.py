import numpy as np, fk, time, ctrl
q0=np.array([0.6772,-0.3431,1.2,-1.4669,1.2422,-1.9544,2.2225])
t=time.time()
n=0
for i in range(200):
    q,ok=fk.ik(np.array([0.5,0.2,0.83]),ctrl.targR(np.pi/2),q0,iters=60); n+=ok
print("ik60 per call ms",(time.time()-t)/200*1000, "ok",n)
