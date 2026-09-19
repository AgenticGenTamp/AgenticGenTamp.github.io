"""Regenerate the downloadable table from the canonical benchmark JSON."""
import csv,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
data=json.loads((root/'data/benchmark.json').read_text())
with (root/'data/benchmark.csv').open('w',newline='') as stream:
    writer=csv.writer(stream,lineterminator="\n")
    writer.writerow(['environment','family','method','backend','access','mean_success','min_run_success','max_run_success'])
    for env in data['environments']:
        for method in data['methods']:
            result=env['results'][method['id']]
            writer.writerow([env['name'],env['family'],method['name'],method['backend'],method['access'],*([result['mean'],result['min'],result['max']] if result else ['','',''])])
print('Updated data/benchmark.csv')
