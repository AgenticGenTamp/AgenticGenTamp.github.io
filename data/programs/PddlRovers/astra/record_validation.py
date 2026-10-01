"""Summarize the final live-server regression runs."""
import json
from pathlib import Path

files = ['validation_final_0_199.txt', 'validation_final_460_659.txt']
rows = []
for filename in files:
    for line in Path(filename).read_text().splitlines():
        parts = line.split()
        if len(parts) == 6 and parts[0] == 'RESULT':
            rows.append(dict(seed=int(parts[1]), objectives=int(parts[2]),
                             steps=int(parts[3]), success=parts[4] == 'True',
                             seconds=float(parts[5])))
summary = dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
               mean_steps=round(sum(r['steps'] for r in rows) / len(rows), 3),
               maximum_steps=max(r['steps'] for r in rows),
               maximum_seconds=max(r['seconds'] for r in rows),
               objective_counts={str(n): sum(r['objectives'] == n for r in rows)
                                 for n in sorted({r['objectives'] for r in rows})},
               failed_seeds=[r['seed'] for r in rows if not r['success']],
               result_files=files)
Path('final_validation.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
