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
  const main = rows.filter(r => r.kind !== 'reference').sort((a, b) => {
    // Methods without coverage always come last, regardless of sort direction.
    if (a.mean === null) return b.mean === null ? 0 : 1;
    if (b.mean === null) return -1;
    return (descending ? -1 : 1) * (a.mean - b.mean);
  });
  const comparable = main.filter(r => r.mean !== null && r.count === r.total).sort((a,b) => b.mean - a.mean);
  const ranks = new Map(comparable.map(r => [r.id, comparable.findIndex(x => Math.abs(x.mean - r.mean) < 1e-10) + 1]));
  return [...main.map(r => ({...r, rank: ranks.get(r.id) ?? null})), ...rows.filter(r => r.kind === 'reference').map(r => ({...r, rank:null}))];
}
