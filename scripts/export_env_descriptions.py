"""Render archived agent descriptions for the static site, preserving original files.

Requires markdown-it-py==3.0.0 only when regenerating this asset. Serving the
website and running the regular checks require no third-party dependencies.
"""
from pathlib import Path
import hashlib
import json

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/environment-descriptions/sources.json'
DESTINATION = ROOT / 'data/environment-descriptions.json'


def main():
    manifest = json.loads(SOURCE.read_text())
    parser = MarkdownIt('commonmark', {'html': False}).enable('table')
    entries = []
    for source in manifest['environments']:
        raw = (ROOT / source['rawFile']).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != source['sha256']:
            raise ValueError(f"Original description changed: {source['id']}")
        text = raw.decode('utf-8')
        # Source headings are nested under the site's section and panel headings.
        # Only markup changes: preserve every word and keep raw Markdown available.
        tokens = parser.parse(text)
        for token in tokens:
            if token.type in ('heading_open', 'heading_close'):
                token.tag = f'h{min(6, int(token.tag[1:]) + 4)}'
        html = parser.renderer.render(tokens, parser.options, {})
        entries.append({'id': source['id'], 'rawFile': source['rawFile'],
                        'sha256': digest, 'html': html})
    DESTINATION.write_text(json.dumps({'setting': manifest['setting'],
                                      'environments': entries},
                                     ensure_ascii=False, indent=2) + '\n')
    print(f'Rendered {len(entries)} archived environment descriptions.')


if __name__ == '__main__':
    main()
