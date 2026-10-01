import hljs from './vendor/highlight/core.min.js';
import python from './vendor/highlight/python.min.js';
hljs.registerLanguage('python', python);

const dialog = document.querySelector('#program-dialog');
const field = id => document.querySelector(`#program-${id}`);
let request, currentClip, currentEnvironment, returnFocus;
let sourceText = '', category = 'final', visibleFiles = [];

function resetReader() {
  request?.abort();
  sourceText = '';
  field('copy').disabled = true;
  field('copy').textContent = 'Copy code';
  field('download').hidden = true;
  field('code').replaceChildren();
  field('lines').textContent = '';
  field('source').scrollTop = 0;
  field('source').scrollLeft = 0;
  field('source').setAttribute('aria-busy', 'false');
}
async function loadFile() {
  resetReader();
  const file = visibleFiles[Number(field('file').value)];
  if (!file) return;
  const controller = new AbortController();
  request = controller;
  field('download').href = file.path;
  field('download').download = `${currentEnvironment.id}-${currentClip.method}-${file.name.replaceAll('/', '-')}`;
  field('download').hidden = false;
  field('status').textContent = 'Loading source…';
  field('source').setAttribute('aria-busy', 'true');
  try {
    const response = await fetch(`${file.path}?v=${file.sha256.slice(0, 12)}`, {signal: controller.signal});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await response.text();
    if (controller.signal.aborted) return;
    sourceText = text;
    // highlight.js escapes source text before adding its syntax markup.
    field('code').innerHTML = hljs.highlight(text, {language: 'python', ignoreIllegals: true}).value;
    field('lines').textContent = Array.from({length: file.lines}, (_, i) => i + 1).join('\n');
    field('status').textContent = `${file.lines.toLocaleString()} lines · Original archived source`;
    field('copy').disabled = false;
  } catch (error) {
    if (error.name !== 'AbortError') field('status').textContent = 'Source could not load. Close and reopen this viewer to retry, or use Download file.';
  } finally {
    if (!controller.signal.aborted) field('source').setAttribute('aria-busy', 'false');
  }
}
function renderFiles() {
  resetReader();
  const files = category === 'final' ? currentClip.program.files : currentClip.program.synthesisFiles;
  const query = field('search').value.trim().toLowerCase();
  visibleFiles = files.filter(file => file.name.toLowerCase().includes(query));
  field('file').replaceChildren(...visibleFiles.map((file, index) => {
    const option = document.createElement('option');
    option.value = index;
    option.textContent = category === 'final' && file.name === 'approach.py' ? 'approach.py (entry point)' : file.name;
    return option;
  }));
  field('file').disabled = !visibleFiles.length;
  field('status').textContent = query ? 'No filenames match your search.' : 'No separate probing scripts are available for this run.';
  if (visibleFiles.length) loadFile();
}
function selectCategory(next) {
  category = next;
  field('search').value = '';
  field('final').setAttribute('aria-pressed', String(category === 'final'));
  field('synthesis').setAttribute('aria-pressed', String(category === 'synthesis'));
  field('purpose').textContent = category === 'final'
    ? 'This is the frozen program used for the video shown here. The same program is evaluated on many held-out instances.'
    : 'Probes, calibration, tests, and intermediate code saved during this run.';
  field('video-note').hidden = category === 'final';
  renderFiles();
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
  field('final').textContent = `Final program (${clip.program.files.length})`;
  field('synthesis').textContent = `Probing & development (${clip.program.synthesisFiles.length})`;
  const omitted = clip.program.synthesisOmittedFiles.length;
  field('file-note').textContent = `${clip.program.files.length} final-program file${clip.program.files.length === 1 ? '' : 's'} · ${clip.program.synthesisFiles.length} probing and development files` +
    (omitted ? `. ${omitted} archived file${omitted === 1 ? '' : 's'} omitted because of private paths.` : '.');
  document.body.classList.add('program-reader-open');
  dialog.showModal();
  selectCategory('final');
}
field('final').addEventListener('click', () => selectCategory('final'));
field('synthesis').addEventListener('click', () => selectCategory('synthesis'));
field('search').addEventListener('input', renderFiles);
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
