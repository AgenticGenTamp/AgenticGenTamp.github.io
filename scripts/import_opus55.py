"""Audit a local final-results snapshot and import complete five-run environment sets.

Usage: python3 scripts/import_opus55.py PATH_TO_FINAL_RESULTS
Requires PyYAML. Source runs are read-only; public provenance uses relative paths.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [24, 42, 222, 424, 444]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def audit(root):
    mapping = json.loads((ROOT / 'data/environment-descriptions/sources.json').read_text())
    entries = []
    used_dirs = set()
    for env in mapping['environments']:
        directories = sorted(root.glob(env['environmentKey'] + '__agentic__none__blackbox__strict__claude_opus55__*'))
        used_dirs.update(directories)
        runs = []
        pending = []
        for directory in directories:
            for run in sorted(directory.glob('*/replicate_*')):
                seed = int(run.name.removeprefix('replicate_'))
                assert seed in SEEDS, run
                path = run / 'results.json'
                if not path.exists():
                    pending.append({'seed': seed, 'directory': str(run.relative_to(root)), 'reason': 'results.json missing'})
                    continue
                result = json.loads(path.read_text())
                config = yaml.safe_load((run / '.hydra/config.yaml').read_text())
                per = result['per_episode']
                assert result['replicate_seed'] == config['replicate_seed'] == seed, run
                assert config['approach']['backend'] == {'backend': 'claude', 'model': 'claude-opus-5-5', 'effort': 'high'}, run
                assert config['approach']['blackbox'] and config['approach']['blackbox_strict'], run
                assert config['approach']['max_budget_usd'] == 20 and config['eval_timeout'] == 60, run
                assert config['eval_seed'] == result['eval_seed'] == 792075, run
                if not result.get('eval_complete') or result['num_evaluated_episodes'] != 100:
                    pending.append({'seed': seed, 'directory': str(run.relative_to(root)), 'reason': 'evaluation incomplete', 'episodes': len(per)})
                    continue
                assert len(per) == result['num_eval_tasks'] == config['num_eval_tasks'] == 100, run
                assert result['num_crashed_episodes'] == 0, run
                assert all(type(e['solved']) is bool for e in per), run
                solved = sum(e['solved'] for e in per)
                assert abs(result['solve_rate'] - solved / 100) < 1e-12, run
                assert result['experiment_id'] == directory.name, run
                runs.append({'seed': seed, 'timestamp': run.parent.name,
                    'member': str(path.relative_to(root)),
                    'programMember': str((run / 'sandbox/approach.py').relative_to(root)),
                    'resultsSha256': digest(path), 'approachSha256': digest(run / 'sandbox/approach.py'),
                    'configSha256': digest(run / '.hydra/config.yaml'),
                    'solved': solved, 'episodes': len(per), 'rate': solved / 100, 'evalSeed': result['eval_seed'],
                    'objectCounts': dict(sorted(Counter(str(e.get('object_count')) for e in per).items())),
                    'costUSD': result.get('agent_cost_usd'), 'synthesisSeconds': result.get('gen_wall_time_s')})
        # Fail on duplicates: silently preferring a revision would hide a changed snapshot.
        assert len({r['seed'] for r in runs}) == len(runs), (env['id'], 'duplicate complete seeds')
        runs.sort(key=lambda r: r['seed'])
        missing = sorted(set(SEEDS) - {r['seed'] for r in runs})
        exact = Decimal(sum(r['solved'] for r in runs)) / 500 if not missing else None
        entries.append({'id': env['id'], 'environmentKey': env['environmentKey'],
            'result': {'mean': float(exact.quantize(Decimal('.01'), rounding=ROUND_HALF_UP)),
                       'min': min(r['rate'] for r in runs), 'max': max(r['rate'] for r in runs)} if exact is not None else None,
            'unroundedMean': float(exact) if exact is not None else None,
            'missingSeeds': missing, 'pendingRuns': pending, 'runs': runs})
    other = []
    for directory in sorted(root.glob('*__claude_opus55__*')):
        if directory in used_dirs:
            continue
        other.append({'directory': directory.name,
                      'reason': '+ source setting' if '__whitebox__' in directory.name else 'Outside the website environment set',
                      'resultFiles': len(list(directory.glob('*/replicate_*/results.json'))),
                      'runDirectories': len(list(directory.glob('*/replicate_*')))})
    return {'method': 'opus55', 'backend': 'Claude Code with Opus 5.5 (high)', 'setting': 'Main setting',
        'snapshotDate': datetime.now(timezone.utc).date().isoformat(),
        'collection': 'Local robocode_final_results snapshot',
        'selection': 'Strict black box, Claude Opus 5.5 high, $20 budget, 60-second evaluation timeout. Require exactly the five expected seeds and 100 complete, crash-free held-out episodes per run. No measured five-run mean is emitted for incomplete environment sets. Reject duplicate complete seeds.',
        'precision': 'Success counts from per_episode; five-run means rounded half up to two decimals, matching the existing table. Unrounded means retained separately.',
        'runsPerEnvironment': 5, 'heldOutInstancesPerRun': 100, 'replicateSeeds': SEEDS,
        'completeRuns': sum(len(e['runs']) for e in entries),
        'evaluationEpisodes': sum(100 * len(e['runs']) for e in entries),
        'includedRuns': sum(5 for e in entries if e['result'] is not None),
        'completeEnvironments': sum(e['result'] is not None for e in entries),
        'environments': entries, 'excludedCollections': other}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_root', type=Path)
    args = parser.parse_args()
    snapshot = audit(args.results_root.resolve())
    data = json.loads((ROOT / 'data/benchmark.json').read_text())
    for env, entry in zip(data['environments'], snapshot['environments']):
        assert env['id'] == entry['id']
        env['results']['opus55'] = entry['result']
    data['coverage'] = {'opus55': {k: snapshot[k] for k in ('completeRuns', 'includedRuns', 'completeEnvironments', 'snapshotDate')}}
    data['coverage']['opus55']['pending'] = [{'id': e['id'], 'name': next(x['name'] for x in data['environments'] if x['id'] == e['id']), 'missingSeeds': e['missingSeeds'], 'completedRuns': len(e['runs'])} for e in snapshot['environments'] if e['missingSeeds']]
    save(ROOT / 'data/opus55-results.json', snapshot)
    save(ROOT / 'data/benchmark.json', data)
    for e in snapshot['environments']:
        print(f"{e['id']}: {len(e['runs'])}/5; missing {e['missingSeeds']}")
    print(f"{snapshot['completeRuns']}/140 complete runs; {snapshot['completeEnvironments']}/28 full environment sets")

if __name__ == '__main__':
    main()
