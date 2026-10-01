import sys; sys.path.insert(0,"scratch"); from h import *
# (a) held block lowered onto table
o,info=env.reset(seed=0); o,_=goto(o,BASE,rstate(o)[3:10]); B=blocks(o)
o,ok,y,r=pick(o,B[1]); print("pick",ok,r, "held z",bpose(o,B[1])[2])
p=bpose(o,B[1])
for z in np.arange(0.94,0.70,-0.002):
    s=rstate(o); qn,e=ikp(BASE,[-0.15,0.15,z],y,s[3:10]); o2,rej=goto(o,BASE,qn)
    if rej: print("(a) held lowering rej at tool z",round(z,3),"block z now",bpose(o2,B[1])[2].round(4),"tool",tool(o2)[0].round(3)); break
    o=o2
o,rej=move_tool(o,BASE,[-0.15,0.15,0.95],y)
# (b) place A at (-0.05,0) then sweep held? we hold B[1]; put it at (-0.05,0) as A
o,rej=carry(o,B[1],(-0.05,0.0),0.0); o=grip(o,1); A=B[1]
# hold B[2]? far. Use B[1] again as A and B[0] w/ base B2
B2=(-0.45,-0.2,0.0)
o,r=goto(o,B2,rstate(o)[3:10]); o,ok,y,r=pick(o,B[0],base=B2); print("pickB",ok,r)
o,rej=carry(o,B[0],(0.1,0.0),0.0,base=B2)
o,rej=carry(o,B[0],(0.1,0.0),0.0,z=0.80,base=B2); print("lowered B at x=0.1 rej",rej,bpose(o,B[0])[:3].round(3))
for x in np.arange(0.1,-0.06,-0.002):
    s=rstate(o); t,R=tool(o); ty=np.arctan2(R[1,1],R[0,1]); qn,e=ikp(B2,[x,0,0.80],ty,s[3:10]); o2,rej=goto(o,B2,qn)
    if rej: print("(b) held block sweep into A rej at block x",bpose(o,B[0])[0].round(4),"(A at -0.05, contact at 0.02)"); break
    o=o2
# (d) teleport held block through A in one step at low height
s=rstate(o); t,R=tool(o); ty=np.arctan2(R[1,1],R[0,1])
qn,e=ikp(B2,[-0.15,0,0.80],ty,s[3:10]); a=np.zeros(11,np.float32); a[3:10]=qn-s[3:10]
print("maxdelta",np.abs(a).max().round(3)); o2=step(o,a); print("(d) one-step jump through A: block now",bpose(o2,B[0])[:3].round(3))
