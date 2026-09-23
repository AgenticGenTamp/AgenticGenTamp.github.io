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
# Freeze the assets actually reviewed in the environment audit.
audit=json.loads((ROOT/'data/environment-audit.json').read_text())
assert audit['paperSha256']==data['source']['sha256']
assert {e['id'] for e in audit['environments']}=={e['id'] for e in data['environments']}
for entry in audit['environments']:
    env=next(e for e in data['environments'] if e['id']==entry['id'])
    assert entry['video']==env['video']
    assert hashlib.sha256((ROOT/env['video']).read_bytes()).hexdigest()==entry['videoSha256']
    assert hashlib.sha256((ROOT/env['poster']).read_bytes()).hexdigest()==entry['posterSha256']
    assert all(entry['checks'].values())
# Policy comparisons must use the same held-out instance, with verifiable results.
examples=json.loads((ROOT/'data/policy-examples.json').read_text())
assert len({e['id'] for e in examples['environments']})==len(examples['environments'])
assert {e['id'] for e in examples['environments']}=={e['id'] for e in data['environments']}
for entry in examples['environments']:
    assert entry['id'] in {e['id'] for e in data['environments']}
    available={v['method'] for v in entry['videos']}
    unavailable={m['method'] for m in entry.get('unavailable',[])}
    assert available|unavailable=={'claude','codex','genplan'} and not available&unavailable
    assert 1<=len(entry['videos'])<=3 and len(available)==len(entry['videos'])
    assert all(m['reason'] and m['label'] for m in entry.get('unavailable',[]))
    # Synthesis seeds can differ across methods; the held-out instance must match.
    for field in ('instanceSeed','episode'):
        assert len({v['source'][field] for v in entry['videos']})==1
    # Exact state hashes allow harmless rasterization differences in initial frames.
    state_hashes=[v['source'].get('initialStateSha256') for v in entry['videos']]
    assert (all(state_hashes) and len(set(state_hashes))==1) or len({v['source']['initialFrameSha256'] for v in entry['videos']})==1
    for clip in entry['videos']:
        assert isinstance(clip['source']['replicateSeed'],int)
        if clip['method']=='codex' and entry['id'] in examples.get('codexRerunEnvironments',[]):
            assert clip['source']['collection']=='Codex Reruns'
        require(clip['video']);require(clip['poster'])
        assert hashlib.sha256((ROOT/clip['video']).read_bytes()).hexdigest()==clip['source']['videoSha256']
        assert clip['solved']==clip['source']['archivedSolved']
        assert isinstance(clip['steps'],int) and clip['steps']>0
        assert all(re.fullmatch(r'[a-f0-9]{64}',clip['source'][key]) for key in ('resultsSha256','approachSha256','initialFrameSha256'))
        if clip['source'].get('initialStateSha256'):
            assert re.fullmatch(r'[a-f0-9]{64}',clip['source']['initialStateSha256'])
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
    audited=next(e for e in audit['environments'] if e['id']==entry['id'])
    assert digest==audited['descriptionSha256'] and entry['environmentKey']==audited['environmentKey']
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
print('PASS: HTML references, 28 audited environments and archived descriptions, 6 methods, matched policy-example provenance, paper checksum, gallery provenance, media sizes, private-path scan.')
