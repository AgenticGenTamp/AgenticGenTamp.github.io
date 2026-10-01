import json
exec(open('calib1.py').read().split("g,err=")[0])
json.dump(obs.tolist(),open('st_pre.json','w'))
g,err=kin.ik_arm(to_arm(cp+[0,0,0.05],base),Rt,pre)
obs=goto(g,0,obs,200)
print('fk g',fkw(obs),'obj',obs[obj:obj+3])
json.dump(obs.tolist(),open('st_g.json','w'))
