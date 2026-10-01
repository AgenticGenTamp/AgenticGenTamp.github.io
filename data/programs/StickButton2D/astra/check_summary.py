import json,sys
for path in sys.argv[1:]:
 records=[json.loads(x) for x in open(path) if x.strip()]
 print(path,'n',len(records),'success',sum(x['success'] for x in records),'mean_steps',round(sum(x['steps'] for x in records)/max(1,len(records)),2),'failures',[(x.get('seed'),x.get('count')) for x in records if not x['success']])
