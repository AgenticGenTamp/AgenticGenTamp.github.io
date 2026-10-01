import sys; sys.path.insert(0,'scratch')
from gm_lib import *
r=R(27,(-0.47,0.22,0.0))
bc,byaw=block(r.o); c,s=math.cos(byaw),math.sin(byaw)
az=np.array([s,-c,0.])
pn=np.array([bc[0],bc[1],0.80]); pf=pn-0.16*az
print(r.move_tool([pf[0],pf[1],0.95],byaw), fk(r.base,r.cfg())[0].round(3))
print(r.move_tool(pf,byaw), fk(r.base,r.cfg())[0].round(3))
for k in range(16,-1,-1):
    ok=r.try_tool(pn-0.01*k*az,byaw); p,Rm=fk(r.base,r.cfg())
    print(k,ok,p.round(3),Rm[:,0].round(2),block(r.o)[0].round(3))
