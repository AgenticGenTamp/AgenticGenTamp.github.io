import urllib.request
url='https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/kinova_gen3/gen3.xml'
try:
 data=urllib.request.urlopen(url,timeout=15).read().decode()
 open('public_gen3.xml','w').write(data)
 print(data[:16000])
except Exception as ex:print(ex)
