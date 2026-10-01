exec(open('probe_vertical.py').read().split('E=make_env()')[0])
for q4 in [-1.8,-1.6,-1.4]:
 def qs(q2):return [0,q2,-np.pi,q4,0,q2-q4-np.pi,np.pi/2]
 q2=brentq(lambda q2:fk(qs(q2))[2]-.145,-.5,1.4);q=qs(q2)
 print(q4,q,fk(q),flush=True)
