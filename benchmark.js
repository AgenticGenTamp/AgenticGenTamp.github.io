// All averages use identical environment weights; absent results remain absent.
export function selectEnvironments(data, scope = 'all', family = 'all') {
  return data.environments.filter(e => (scope !== 'shared' || e.results.planner !== null) && (family === 'all' || e.family === family));
}
export function summarize(data, scope = 'all', family = 'all', descending = true) {
  const environments = selectEnvironments(data, scope, family);
  const rows = data.methods.map(method => {
    const available = environments.map(e => e.results[method.id]).filter(r => r !== null);
    return {...method, count: available.length, total: environments.length, mean: available.length ? available.reduce((sum, r) => sum + r.mean, 0) / available.length : null};
  });
  const compareScores = (a, b) => {
    // Methods without coverage always come last, regardless of sort direction.
    if (a.mean === null) return b.mean === null ? 0 : 1;
    if (b.mean === null) return -1;
    return (descending ? -1 : 1) * (a.mean - b.mean);
  };
  const main = rows.filter(r => r.kind !== 'reference').sort(compareScores);
  const references = rows.filter(r => r.kind === 'reference').sort(compareScores);
  const comparable = main.filter(r => r.mean !== null && r.count === r.total).sort((a,b) => b.mean - a.mean);
  const ranks = new Map(comparable.map(r => [r.id, comparable.findIndex(x => Math.abs(x.mean - r.mean) < 1e-10) + 1]));
  return [...main.map(r => ({...r, rank: ranks.get(r.id) ?? null})), ...references.map(r => ({...r, rank:null}))];
}

// Count generated programs, not planner evaluations. Legacy paper methods have
// table coverage and a recorded run protocol; newer methods have individual runs.
export function experimentTotals(data, resultsByMethod) {
  const environmentIds = new Set(data.environments.map(e => e.id));
  const paperPrograms = data.protocol.paperMethods.reduce((sum, id) =>
    sum + data.environments.filter(e => e.results[id] != null).length * data.protocol.runs, 0);
  let programs = paperPrograms;
  let episodes = paperPrograms * data.protocol.heldOutInstances;
  const additionalByBackend = new Map();
  for (const method of data.methods.filter(m => m.resultSource)) {
    if (data.protocol.paperMethods.includes(method.id)) throw new Error(`Duplicate paper/run source: ${method.id}`);
    const source = resultsByMethod[method.id];
    if (!source || source.method !== method.id) throw new Error(`Missing or mismatched run source: ${method.id}`);
    const runs = source.environments.filter(e => environmentIds.has(e.id)).flatMap(e => e.runs);
    programs += runs.length;
    episodes += runs.reduce((sum, run) => sum + run.episodes, 0);
    additionalByBackend.set(method.backend, (additionalByBackend.get(method.backend) || 0) + runs.length);
  }
  return {environments: environmentIds.size, programs, episodes,
    additional: [...additionalByBackend].map(([backend, programs]) => ({backend, programs}))};
}
