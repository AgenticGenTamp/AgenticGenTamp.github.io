import sys; sys.path.insert(0,"scratch"); from h import *
o,info=env.reset(seed=1); o,_=goto(o,BASE,rstate(o)[3:10])
b0,b1=blocks(o)
o,ok,_,r=pick(o,b0); o,rej=carry(o,b0,(-0.03,0.06),0.0); o=grip(o,1)
p=bpose(o,b1); y=byaw(o,b1); y=y-np.round(y/(np.pi/2))*np.pi/2
o,r1=move_tool(o,BASE,[p[0],p[1],0.95],y); print("above",r1,tool(o)[0].round(3),rstate(o)[3:10].round(2))
for z in np.arange(0.94,0.79,-0.01):
    s=rstate(o); qn,err=ikp(BASE,[p[0],p[1],z],y,s[3:10])
    o,r=goto(o,BASE,qn)
    if r: print("rej at",z,"err",err,"q",s[3:10].round(2),"qn",qn.round(2)); break
