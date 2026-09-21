# AgenticGenTAMP project website

The formal project website for **Coding Agents for Generalized Task and Motion Planning Problems**. This is a static GitHub Pages site with no build step or runtime dependencies.

**Deployment repository:** https://github.com/AgenticGenTamp/AgenticGenTamp.github.io

**Website:** https://AgenticGenTamp.github.io/

## Run locally

From this directory:

```sh
python3 serve.py 8770
```

Open http://127.0.0.1:8770/. The included server supports byte ranges for seeking in videos. Use HTTP instead of opening `index.html` as a file: the page fetches JSON and loads JavaScript modules.

## Page structure

- Header and title: the PRPL logo, the full paper title, a background rollout mosaic, and matching Paper, Code, Leaderboard, and Gallery links.
- Setup: a definition of generalized task and motion planning, three instances of the same Obstruction task, and the synthesis and held-out evaluation protocol.
- Gallery: examples near the top of the page. Clips loop silently when visible, with native playback controls. Captions describe the behavior and identify the coding agent and the paper's **Main setting / + source** terminology. Seeds and episode identifiers remain in the data, but are not displayed on cards.
- Project video: a normal MP4 with native controls, a fixed 16:9 frame, and a download link. A short paragraph identifies the Shelf investigation as an observed example, without presenting it as a prescribed pipeline.
- Leaderboard: all six methods from the paper, with environment-family and matched-planner-coverage controls. Click the score heading to reverse sorting.
- Environment explorer: all 28 environments, all six method rows, and printed run-level ranges, followed by source-access and computation-time observations.

The background and gallery videos pause off screen or when the tab is hidden. Reduced-motion and data-saving preferences disable their automatic playback. Manually paused gallery clips stay paused when returning to them. The main video plays only on request and pauses other visible gallery clips while playing.

## Sources and result definitions

`assets/paper.pdf` is the manuscript supplied on 19 September 2026. Its SHA-256 is recorded in `data/benchmark.json`. It remains the original manuscript, including its existing links; no author list or arXiv identifier has been invented.

All success rates and run-level min/max values are transcribed from **Tables I–II**, checked against the rendered table page. Equal-environment averages use these printed two-decimal entries, so aggregates can differ slightly from averages of unrounded source logs. A zero is a measured failure rate; `null` means no planner was provided.

- All-environment scope: 28 environments; the planner covers 16 and is shown without a rank.
- Planner scope: exactly the same 16 environments for every method.
- Source-access reference: always shown separately, unranked, because its interface differs from the main setting.
- Min/max in the explorer are across five runs, not confidence intervals.
- Table III computation times use 44 matched seeds across 13 environments where both settings reach 100% success. They are not all-environment averages.

### Media and branding

`film/` is vendored from https://github.com/merlerm/agentamp-video at commit `d81f2c1`. The original HTML, CSS, JavaScript, local fonts, stills, and required clips are retained as presentation source material. Local-path metadata JSON and development scripts are excluded. The page no longer embeds the HTML player.

`assets/project-video.mp4` is an export of that presentation in recording mode, with the default Shelf example and captions disabled. The presentation's original on-screen explanations, speed labels, and waits for unfinished clips are preserved. It is a silent, 1280 × 720, 25 fps H.264 video, encoded with CRF 23, YUV 4:2:0, and MP4 fast-start. Its duration is 3 minutes 26.2 seconds. The recording was checked for media-loading errors and sampled across all scenes. The poster is a frame from this export.

To export a future revision, serve `film/` over HTTP, record its 1280 × 720 stage using `?record=1&theme=poster`, and let its clock finish, including clip-completion holds. Use the recording helpers in the upstream presentation repository as a starting point. Hide the authoring controls and use the default on-screen explanations; the optional narration captions are a draft. Convert the recording to H.264 MP4 with `-pix_fmt yuv420p -movflags +faststart`, and keep the file under GitHub's 100 MiB per-file limit. Replace both the MP4 and its poster, then check playback and seeking.

The hero and gallery posters are derived from the supplied clips. Setup stills show Obstruction3D run 24, episodes 26, 92, and 84 (one, two, and four obstacles). Gallery provenance is recorded in `data/gallery.json`; media names retain the upstream run and episode identifiers.

`assets/prpl-robot.png` is the unmodified PRPL robot logo from https://prpl-group.com/assets/images/prpl-robot.png. It links to the Princeton Robot Planning and Learning lab website.

## Update content

### Paper and links

Replace `assets/paper.pdf`, update the SHA-256 and version in `data/benchmark.json`, and refresh table data if the manuscript changes. Paper links are local. Research-code links point to https://github.com/tomsilver/robocode; edit the links in `index.html` if the team moves it. Website-source links point to this formal repository.

### Leaderboard

Edit `data/benchmark.json`. Every environment must include one result for each method ID, using `{ "mean": 0.74, "min": 0.50, "max": 0.99 }` or `null`. Run `python3 scripts/export_csv.py` so the downloadable table matches the interactive one. Keep the evaluation protocol and source version explicit. Do not mix newer partial reruns into a complete published table without identifying their coverage.

### Gallery

Add an MP4 to `film/assets/clips/`, a JPEG poster to `assets/posters/`, and an entry to `data/gallery.json`. The `file` value is the common basename without the extension. Include `id`, `title`, `description`, `category`, `environment`, `method`, `backend`, `setting`, `seed`, and `episode`. Use `Main setting` or `+ source` for `setting`. Supported categories are `Strategy`, `Tool use`, and `Recovery`. Clips play at 1× in the gallery; the project video uses its own labeled playback speeds.

Keep titles descriptive and explain the observed behavior in plain language. Seeds, episode numbers, and method provenance should remain in the data even though the page shows a shorter caption.

The formal site is independent of the earlier anonymous gallery's Drive sync process. It does not change or invoke that pipeline.

## Verify

With Node.js and Python 3 available:

```sh
npm test
npm run check
```

The checks cover comparison scopes, missing-vs-zero data, aggregate ranking, all filter combinations, run ranges, local assets, the paper checksum, and accidental private paths. Review desktop/mobile layout and video playback in a browser before publishing visual changes.

## Deploy

GitHub Pages serves the `main` branch at the repository root (`/`). `.nojekyll` disables Jekyll processing. No workflow or build service is needed.

Before pushing, verify the destination:

```sh
git remote get-url origin
# Must be https://github.com/AgenticGenTamp/AgenticGenTamp.github.io.git
git push origin main
```

Never run the anonymous site's publication scripts from this repository.
