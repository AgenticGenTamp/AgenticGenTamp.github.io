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

- Full-screen background: a compressed copy of the 240-rollout mosaic from the supplied video materials. It pauses off screen and respects reduced-motion and data-saving preferences.
- Project film: the original time-driven HTML presentation, embedded on demand. Chapter buttons open its timeline directly. The supplied material did not contain a single exported main MP4; the original playable presentation is retained without changing its scenes. Use `film/` to open it separately.
- Method: the Shelf calibration trace, with three selectable stages and original clips.
- Leaderboard: all six rows from the paper, with environment-family and matched-planner-coverage controls. Click the score heading to reverse sorting.
- Environment explorer: all 28 environments, all six method rows, and printed run-level ranges.
- Gallery: large inline videos that loop silently as they enter the viewport, with native playback controls and method, backend, setting, synthesis seed, and episode. Offscreen and hidden-tab videos pause; reduced-motion and data-saving preferences disable automatic playback. Manually paused clips stay paused when returning to them.

## Sources and result definitions

`assets/paper.pdf` is the manuscript supplied on 19 September 2026. Its SHA-256 is recorded in `data/benchmark.json`. It remains the original manuscript, including its existing links; no author list or arXiv identifier has been invented.

All success rates and run-level min/max values are transcribed from **Tables I–II**, checked against the rendered table page. Equal-environment averages use these printed two-decimal entries, so aggregates can differ slightly from averages of unrounded source logs. A zero is a measured failure rate; `null` means no planner was provided.

- All-environment scope: 28 environments; the planner covers 16 and is shown without a rank.
- Planner scope: exactly the same 16 environments for every method.
- Source-access reference: always shown separately, unranked, because its interface differs from the main setting.
- Min/max in the explorer are across five runs, not confidence intervals.
- Table III computation times use 44 matched seeds across 13 environments where both settings reach 100% success. They are not all-environment averages.

`film/` is vendored from https://github.com/merlerm/agentamp-video at commit `d81f2c1`. The original HTML, CSS, JavaScript, local fonts, stills, and required clips are retained. Local-path metadata JSON and development scripts are excluded. The only player integration change accepts a same-origin pause message from the parent. The hero and gallery posters are derived from those supplied clips. Prominent Leaderboard and Gallery buttons sit alongside Paper and Code below the main title. Gallery provenance is recorded in `data/gallery.json`; media names retain the upstream run and episode identifiers.

## Update content

### Paper and links

Replace `assets/paper.pdf`, update the SHA-256 and version in `data/benchmark.json`, and refresh table data if the manuscript changes. Paper links are local. Research-code links currently point to the public research repository https://github.com/tomsilver/robocode; edit the two links in `index.html` if the team moves it. Website-source links point to this formal repository.

### Leaderboard

Edit `data/benchmark.json`. Every environment must include one result for each method ID, using `{ "mean": 0.74, "min": 0.50, "max": 0.99 }` or `null`. Run `python3 scripts/export_csv.py` so the downloadable table matches the interactive one. Keep the evaluation protocol and source version explicit. Do not mix newer partial reruns into a complete published table without identifying their coverage.

### Gallery

Add an MP4 to `film/assets/clips/`, a JPEG poster to `assets/posters/`, and an entry to `data/gallery.json`. The `file` value is the common basename without the extension. Include `id`, `title`, `description`, `category`, `environment`, `method`, `backend`, `setting`, `seed`, and `episode`. Supported categories are `Strategy`, `Tool use`, and `Recovery`. Clips play at 1× in the gallery; the original film sets its own labeled playback speeds.

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
