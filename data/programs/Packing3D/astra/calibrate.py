exec(open('probe_fk.py').read().split('t=np.eye')[0])
q=[0,.391676098,-np.pi,-2.05185294,0,-.452491313,np.pi/2]
t=np.eye(4)
for (xyz,rx),qi in zip(params,q):
 a=np.eye(4);a[:3,3]=xyz;a[:3,:3]=R.from_euler('x',rx).as_matrix()@R.from_euler('z',qi).as_matrix();t=t@a
obj=np.array([.312654048,.288928896,.095]);tf=np.array([-.0003475547,-.0322969258,.0474185422]);rot=R.from_quat([.701782584,.701783717,.086605251,-.086604618]).inv().as_matrix()
ee=obj-rot@tf
print('Actual EE world',ee,'rot',rot)
print('Model last joint local',t)
print('Offset in model frame',t[:3,:3].T@(ee-np.array([-.22,.287928909,0])-t[:3,3]))
print('Rot offset',t[:3,:3].T@rot)
