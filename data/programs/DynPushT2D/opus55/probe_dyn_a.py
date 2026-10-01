from probe_dyn_lib import *
S=Sim()
# pick seed with central block
o=S.reset(3); print(o[0:3],o[12:15])
w,lh,lv=o[12:15]
moved=S.goto_local([0,-lv-0.1-0.15]); print('moved during nav',moved)
print('robot local',w2l(S.obs,S.obs[16:18]))
R=S.push([0,1],0.03,25)
for r in R: print(r[:3], 'rp',r[3:5],'rp2',r[5:7])
