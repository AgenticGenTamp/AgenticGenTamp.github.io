from exprot_4 import *
o=replay(seed,acts)
for i in range(8): o=step(o,dth=-math.pi/16)
o=goto(o,1.2,1.006); o=goto(o,1.2,1.15); r,s,_=rd(o); print('pre-button robot',round(r['x'],3),round(r['y'],3),bb(s))
pressed=False
for i in range(40):
    r,s,b=rd(o); o2=step(o,dth=0.02); r2,s2,b2=rd(o2)
    acc=abs(r2['theta']-r['theta'])>1e-7
    bb0=[x for x in b2 if x[0]=='button0'][0]; d=dist(s2,2.113,1.801)
    if d<0.12 or not acc: print('  rot +0.02',('ACC' if acc else 'REJ'),'th %.3f dist-to-b0 %.4f (overlap if <0.05)'%(r2['theta'],d),'b0 color r,g',round(bb0[3],2),round(bb0[4],2))
    o=o2
    if bb0[4]>0.5 and d>0.12: print('  passed; still pressed after leaving:',round(bb0[4],2)); break
