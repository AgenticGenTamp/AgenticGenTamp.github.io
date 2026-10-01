from env_client import make_env
for count in [0,5,7,8,9]:
 e=make_env()
 try:
  s,i=e.reset(seed=0,options={'object_count':count})
  print(count,'accepted',i)
 except Exception as exc: print(count,str(exc))
 finally:e.close()
