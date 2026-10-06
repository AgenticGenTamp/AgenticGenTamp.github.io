s=open('scratch/ik_grasp.py').read()
s=s.replace('can=objpos("can_0"); print','''for _ in range(15):
    b=np.array(re.qpos["base"]); a=np.zeros(11); a[:3]=np.clip(np.array([-0.05,-0.25,0])-b,-0.1,0.1); env.step(a)
print("base now",np.round(np.array(re.qpos["base"]),3))
can=objpos("can_0"); print''')
s=s.replace('def goto(qt, grip, n=40','def goto(qt, grip, n=100')
s=s.replace('print("steps pre",goto(q1,0))','print("steps pre",goto(q1,0,tol=0.005))')
s=s.replace('q2,e=ik(can+[0,0,0.0],q1); print("steps down",goto(q2,0)','q2,e=ik(can+[0,0,0.06],q1); goto(q2,0,tol=0.005); q2,e=ik(can+[0,0,0.0],q2); print("steps down",goto(q2,0,tol=0.005)')
open('scratch/ik_grasp2.py','w').write(s)
