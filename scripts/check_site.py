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
    if p.suffix in ('.js','.css','.html','.json'):
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
for g in json.loads((ROOT/'data/gallery.json').read_text()):
    require(f"film/assets/clips/{g['file']}.mp4");require(f"assets/posters/{g['file']}.jpg")
    assert all(key in g for key in ('method','backend','setting','seed','episode','description'))
require('assets/hero.mp4')
if errors:raise SystemExit('\n'.join(errors))
print('PASS: HTML references, 28 environments, 6 methods, paper checksum, gallery provenance, media sizes, private-path scan.')
