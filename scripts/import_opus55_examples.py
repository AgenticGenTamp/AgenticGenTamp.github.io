"""Import verified replay media and byte-for-byte program sources.

Run with the robocode Python environment:
  python scripts/import_opus55_examples.py RESULTS_ROOT RENDER_DIRECTORY
RENDER_DIRECTORY contains <environment>/<environment>.{json,mp4,jpg} and
initial-states.json, independently comparing archived Opus and Astra resets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from robocode.utils.strict_blackbox import reachable_sibling_files

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_root', type=Path)
    parser.add_argument('render_directory', type=Path)
    args = parser.parse_args()
    result_root = args.results_root.resolve()
    renders = args.render_directory.resolve()
    audit = json.loads((ROOT / 'data/opus55-results.json').read_text())
    examples = json.loads((ROOT / 'data/policy-examples.json').read_text())
    states = json.loads((renders / 'initial-states.json').read_text())
    assert {s['id'] for s in states} == {e['id'] for e in examples['environments']}
    for entry in examples['environments']:
        env_id = entry['id']
        result = next(e for e in audit['environments'] if e['id'] == env_id)
        run = next(r for r in result['runs'] if r['seed'] == 42)
        state = next(s for s in states if s['id'] == env_id)
        astra = next(v for v in entry['videos'] if v['method'] == 'astra')
        meta = json.loads((renders / env_id / f'{env_id}.json').read_text())
        assert meta['match'] and meta['archivedSolved'] == meta['policyPassSolved'] == meta['replaySolved'], env_id
        for key in ('resultsSha256', 'approachSha256'):
            assert meta[key] == run[key], (env_id, key)
        assert state['opus55ResultsSha256'] == run['resultsSha256']
        assert state['astraResultsSha256'] == astra['source']['resultsSha256']
        for key in ('episode', 'instanceSeed'):
            assert meta[key] == state[key] == astra['source'][key], (env_id, key)
        assert meta['objectCount'] == state['objectCount']
        assert meta['frameStride'] == 3 and meta['replicateSeed'] == 42
        sandbox = (result_root / run['programMember']).parent
        assert digest(sandbox / 'approach.py') == run['approachSha256']
        assert digest(result_root / run['member']) == run['resultsSha256']
        final = reachable_sibling_files(sandbox / 'approach.py')
        final = [sandbox / 'approach.py'] + sorted(p for p in final if p != sandbox / 'approach.py')
        all_sources = [p for p in sorted(sandbox.rglob('*.py')) if not any(part.startswith('.') for part in p.relative_to(sandbox).parts)]
        development = [p for p in all_sources if p not in final and p.name not in {'env_client.py', 'test_approach.py'}]
        program = {'files': [], 'synthesisFiles': [], 'synthesisArchiveChecked': True, 'synthesisOmittedFiles': []}
        for key, paths in [('files', final), ('synthesisFiles', development)]:
            for path in paths:
                name = path.relative_to(sandbox).as_posix()
                raw = path.read_bytes()
                if re.search(r'/home/|/Users/|AIza[\w-]{30}|gh[pousr]_[\w]{25}', raw.decode('utf-8-sig')):
                    assert key != 'files', (env_id, name, 'private path in required final source')
                    program['synthesisOmittedFiles'].append(name)
                    continue
                dest = ROOT / 'data/programs' / env_id / 'opus55' / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(raw)
                program[key].append({'name': name, 'path': dest.relative_to(ROOT).as_posix(),
                    'sha256': hashlib.sha256(raw).hexdigest(), 'lines': len(raw.decode('utf-8-sig').splitlines()),
                    'archiveMember': path.relative_to(result_root).as_posix()})
        for suffix in ('mp4', 'jpg'):
            shutil.copy2(renders / env_id / f'{env_id}.{suffix}', ROOT / 'assets/policies' / f'{env_id}-opus55.{suffix}')
        assert digest(ROOT / 'assets/policies' / f'{env_id}-opus55.mp4') == meta['videoSha256']
        astra['source']['initialObservationSha256'] = state['initialObservationSha256']
        source = {'kind': 'local-experiment', 'collection': audit['collection'],
                  'resultsMember': run['member'], 'programMember': run['programMember'],
                  'replicateSeed': 42, 'initialObservationSha256': state['initialObservationSha256'],
                  **{key: meta[key] for key in ('episode', 'instanceSeed', 'objectCount', 'resultsSha256', 'approachSha256', 'archivedSolved', 'archivedSteps', 'initialFrameSha256', 'videoSha256', 'replaySolved', 'replaySteps', 'policyPassSolved', 'policyPassSteps', 'frameStride', 'singlePass')}}
        entry['videos'] = [v for v in entry['videos'] if v['method'] != 'opus55'] + [{
            'method': 'opus55', 'label': 'Claude Code with Opus 5.5 (high)', 'setting': 'Main setting',
            'video': f'assets/policies/{env_id}-opus55.mp4', 'poster': f'assets/policies/{env_id}-opus55.jpg',
            'solved': meta['replaySolved'], 'steps': meta['replaySteps'], 'source': source, 'program': program}]
        order = ['claude', 'codex', 'genplan', 'astra', 'opus55']
        entry['videos'].sort(key=lambda v: order.index(v['method']))
        print(env_id, len(final), 'final files,', len(program['synthesisFiles']), 'development files,', len(program['synthesisOmittedFiles']), 'omitted')
    examples['opus55Selection'] = 'Opus 5.5 high uses synthesis seed 42 and the existing shared episode in each environment, including completed Rovers seed 42. Compare canonical initial observations against the archived Astra configuration and instance; verify policy and replay outcomes against the original results. Media is sampled every three actions at 10 fps with a one-second final hold. Rovers now has all five completed runs in the aggregate results. Final source and sibling modules are copied byte for byte. Probing/development files exclude provided clients, hidden files, and files containing private paths; omitted filenames are recorded. File presence does not imply execution.'
    examples['playback'] = 'Frames sampled every 3 actions, encoded at 10 fps (30 actions per video second), with a one-second final hold. Same playback rate for all methods; not wall-clock computation time.'
    (ROOT / 'data/policy-examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False) + '\n')
    state_audit = {'serialization': 'SHA-256 of UTF-8 JSON with sorted keys and compact separators. Array observations use tolist(); object-centric observations use a list sorted by object name and type, with name, type, and unrounded feature values. Reset with the recorded instance seed and object count using each original Hydra environment configuration; enable scene backgrounds only.', 'environments': states}
    (ROOT / 'data/opus55-example-audit.json').write_text(json.dumps(state_audit, indent=2, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
