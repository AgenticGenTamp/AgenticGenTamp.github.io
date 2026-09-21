"""Check deployable assets, table precision, and video provenance without dependencies."""
from pathlib import Path
from html.parser import HTMLParser
import json, re, hashlib, csv
ROOT=Path(__file__).resolve().parents[1]
errors=[]
def require(path):
    if not (ROOT/path).is_file(): errors.append(f'Missing: {path}')
class Assets(HTMLParser):
    def __init__(self, parent):super().__init__();self.parent=parent
    def handle_starttag(self, tag, attrs):
        for key,value in attrs:
            if key not in ('src','href','poster','data-src') or not value:continue
            if value.startswith(('https:','http:','#','data:','mailto:')):continue
            target=(self.parent/value.split('?')[0].split('#')[0])
            if not target.exists():errors.append(f'Missing HTML reference: {target.relative_to(ROOT)}')
for page in ROOT.rglob('*.html'):
    Assets(page.parent).feed(page.read_text())
for p in ROOT.rglob('*'):
    if '.git' in p.parts or not p.is_file():continue
    if p.stat().st_size>=100*1024*1024:errors.append(f'Exceeds GitHub file limit: {p.relative_to(ROOT)}')
    if p.suffix in ('.js','.css','.html','.json','.md'):
        if re.search(r'/home/|/Users/|AIza[\w-]{30}|gh[pousr]_[\w]{25}',p.read_text()):errors.append(f'Private path or credential pattern: {p.relative_to(ROOT)}')
data=json.loads((ROOT/'data/benchmark.json').read_text())
assert len(data['environments'])==28 and len(data['methods'])==6
with (ROOT/'data/benchmark.csv').open() as stream:
    csv_rows=list(csv.DictReader(stream))
assert len(csv_rows)==28*6
for index,(env,method) in enumerate((e,m) for e in data['environments'] for m in data['methods']):
    row=csv_rows[index];result=env['results'][method['id']]
    assert row['environment']==env['name'] and row['family']==env['family']
    assert row['method']==method['name'] and row['backend']==method['backend']
    for key,field in [('mean','mean_success'),('min','min_run_success'),('max','max_run_success')]:
        assert (float(row[field]) if row[field] else None)==(result[key] if result else None)
assert hashlib.sha256((ROOT/'assets/paper.pdf').read_bytes()).hexdigest()==data['source']['sha256']
for e in data['environments']:
    require(e['video']);require(e['poster']);assert len(e['results'])==6
# Every paper environment has exactly one original, checksum-verified description.
manifest=json.loads((ROOT/'data/environment-descriptions/sources.json').read_text())
descriptions=json.loads((ROOT/'data/environment-descriptions.json').read_text())
expected_ids={e['id'] for e in data['environments']}
assert len(manifest['environments'])==len(descriptions['environments'])==28
assert {e['id'] for e in manifest['environments']}==expected_ids
assert {e['id'] for e in descriptions['environments']}==expected_ids
rendered={e['id']:e for e in descriptions['environments']}
for entry in manifest['environments']:
    require(entry['rawFile'])
    digest=hashlib.sha256((ROOT/entry['rawFile']).read_bytes()).hexdigest()
    assert digest==entry['sha256']==rendered[entry['id']]['sha256']
    assert entry['rawFile']==rendered[entry['id']]['rawFile']
    assert entry['source']['kind']=='archived-experiment'
    assert entry['source']['matchingArchivedDescriptions']>=5
    assert 'Action' in rendered[entry['id']]['html']
    assert 'Reward' in rendered[entry['id']]['html'] or 'Goal' in rendered[entry['id']]['html']
    assert not re.search(r'<(?:script|iframe|object|embed)\b|\son\w+=', rendered[entry['id']]['html'], re.I)
# Guard the similarly named dynamic and kinematic Shelf environments.
shelf=next(e for e in manifest['environments'] if e['id']=='Shelf3D')
assert shelf['environmentKey']=='dynamicshelf3d_generalized'
assert shelf['source']['environmentImplementation']=='kinder.envs.dynamic3d.task_families:Shelf3DEnv'
base_motion=next(e for e in manifest['environments'] if e['id']=='BaseMotion3D')
assert base_motion['source']['environmentImplementation']=='kinder/BaseMotion3D-v0'
for g in json.loads((ROOT/'data/gallery.json').read_text()):
    require(f"film/assets/clips/{g['file']}.mp4");require(f"assets/posters/{g['file']}.jpg")
    assert all(key in g for key in ('method','backend','setting','seed','episode','description'))
require('assets/hero.mp4')
require('assets/project-video.mp4')
require('assets/prpl-robot.png')
if errors:raise SystemExit('\n'.join(errors))
print('PASS: HTML references, 28 environments and archived descriptions, 6 methods, paper checksum, gallery provenance, media sizes, private-path scan.')
