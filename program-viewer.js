const dialog = document.querySelector('#program-dialog');
const field = id => document.querySelector(`#program-${id}`);
let request, currentClip, currentEnvironment, returnFocus;
let sourceText = '';

function resetReader() {
  request?.abort();
  sourceText = '';
  field('copy').disabled = true;
  field('copy').textContent = 'Copy code';
  field('code').textContent = '';
  field('lines').textContent = '';
  field('source').scrollTop = 0;
  field('source').scrollLeft = 0;
}
async function loadFile() {
  resetReader();
  const file = currentClip.program.files[Number(field('file').value)];
  const controller = new AbortController();
  request = controller;
  field('download').href = file.path;
  field('download').download = `${currentEnvironment.id}-${currentClip.method}-${file.name.replaceAll('/', '-')}`;
  field('status').textContent = 'Loading source…';
  field('source').setAttribute('aria-busy', 'true');
  try {
    const response = await fetch(`${file.path}?v=${file.sha256.slice(0, 12)}`, {signal: controller.signal});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await response.text();
    if (controller.signal.aborted) return;
    sourceText = text;
    // Source stays inert, including strings containing HTML or script tags.
    field('code').textContent = text;
    field('lines').textContent = Array.from({length: file.lines}, (_, i) => i + 1).join('\n');
    field('status').textContent = `${file.lines.toLocaleString()} lines · Original archived source`;
    field('copy').disabled = false;
  } catch (error) {
    if (error.name !== 'AbortError') field('status').textContent = 'Source could not load. Close and reopen this viewer to retry, or use Download file.';
  } finally {
    if (!controller.signal.aborted) field('source').setAttribute('aria-busy', 'false');
  }
}
export function openProgramViewer(environment, clip, trigger) {
  currentEnvironment = environment;
  currentClip = clip;
  returnFocus = trigger;
  field('title').textContent = environment.name;
  field('method').textContent = `${clip.label} · ${clip.setting}`;
  field('outcome').textContent = `${clip.solved ? 'Success' : 'Failure'} · ${clip.steps} actions`;
  field('instance').textContent = `Synthesis seed ${clip.source.replicateSeed} · Held-out episode ${clip.source.episode}`;
  field('video').src = `${clip.video}?v=${clip.source.videoSha256.slice(0, 12)}`;
  field('video').poster = clip.poster;
  field('video').setAttribute('aria-label', `${clip.label} in ${environment.name}, matching program example`);
  field('file').replaceChildren(...clip.program.files.map((file, index) => {
    const option = document.createElement('option');
    option.value = index;
    option.textContent = file.name === 'approach.py' ? 'approach.py (entry point)' : file.name;
    return option;
  }));
  field('file-note').textContent = clip.program.files.length > 1
    ? `Entry point and ${clip.program.files.length - 1} imported Python helper file${clip.program.files.length === 2 ? '' : 's'}.`
    : 'Single Python source file.';
  document.body.classList.add('program-reader-open');
  dialog.showModal();
  loadFile();
}
field('file').addEventListener('change', loadFile);
field('close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  if (event.target !== dialog) return;
  const box = dialog.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
});
dialog.addEventListener('close', () => {
  resetReader();
  field('video').pause();
  field('video').removeAttribute('src');
  field('video').load();
  document.body.classList.remove('program-reader-open');
  returnFocus?.focus({preventScroll: true});
});
field('copy').addEventListener('click', async () => {
  const text = sourceText;
  try {
    await navigator.clipboard.writeText(text);
    if (text === sourceText) field('copy').textContent = 'Copied';
  } catch {
    field('status').textContent = 'Copy unavailable. Select the code to copy it, or download the file.';
  }
});
