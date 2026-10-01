import sys; sys.path.insert(0,"scratch"); from h import *
B2=(-0.45,-0.2,0.0)
def setup():
    o,info=env.reset(seed=0); o,_=goto(o,BASE,rstate(o)[3:10]); B=blocks(o)
    o,ok,y,r=pick(o,B[1]); o,rej=carry(o,B[1],(-0.05,0.0),0.0); o=grip(o,1)
    o,r=goto(o,B2,rstate(o)[3:10]); o,ok,y,r=pick(o,B[0],base=B2)
    return o,B
for tx in [-0.22]:
    o,B=setup()
    o,rej=carry(o,B[0],(0.1,0.0),0.0,base=B2); o,rej=carry(o,B[0],(0.1,0.0),0.0,z=0.80,base=B2)
    s=rstate(o); t,R=tool(o); ty=np.arctan2(R[1,1],R[0,1])
    qn,e=ikp(B2,[tx,0,0.80],ty,s[3:10]); print("ik e",e)
    a=np.zeros(11,np.float32); a[3:10]=qn-s[3:10]; o2=step(o,a)
    print("one-step jump to",tx,"block now",bpose(o2,B[0])[:3].round(3))
    # check target reachable via high path
    o3,r1=carry(o2,B[0],(tx,0.0),0.0,base=B2); o3,r2=carry(o3,B[0],(tx,0.0),0.0,z=0.80,base=B2); print(" via high path rej",r1,r2,bpose(o3,B[0])[:3].round(3))
# Q2b: open + move in same step
o,B=setup()
o,rej=carry(o,B[0],(0.1,0.0),0.0,base=B2); s=rstate(o); t,R=tool(o); ty=np.arctan2(R[1,1],R[0,1])
qn,e=ikp(B2,[0.05,0.05,0.95],ty,s[3:10]); a=np.zeros(11,np.float32); a[3:10]=qn-s[3:10]; a[10]=1
o=step(o,a); print("open+move from (0.1,0) to (0.05,0.05): block",bpose(o,B[0])[[0,1,2,7]].round(3),"tool",tool(o)[0].round(3))
