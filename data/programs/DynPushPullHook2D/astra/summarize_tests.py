from pathlib import Path
import re,json
pattern=re.compile(r"RESULT (\d+) \{'object_count': (\d+)\} success (True|False) steps (\d+)")
results={}
for path in Path('.').glob('*.txt'):
 for match in pattern.finditer(path.read_text(errors='replace')):
  seed,count,success,steps=match.groups();key=(int(seed),int(count));d=results.setdefault(key,{'success':False,'steps':[],'files':[]})
  if success=='True':d['success']=True;d['steps'].append(int(steps));d['files'].append(path.name)
summary={'instances':len(results),'solved':sum(x['success'] for x in results.values()),'unresolved':[list(k) for k,v in results.items() if not v['success']],'max_success_steps':max((min(v['steps']) for v in results.values() if v['steps']),default=0),'tested_counts':sorted({k[1] for k in results})}
print(json.dumps(summary,indent=2))
Path('validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
