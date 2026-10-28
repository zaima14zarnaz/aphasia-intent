#!/usr/bin/env python3
"""Package only web assets and the local videos referenced by the manifest."""
import json
import shutil
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]

def build():
    output = ROOT / 'dist'
    manifest = json.loads((ROOT / 'data/clips.json').read_text(encoding='utf-8'))
    local_files = []
    for clip in manifest['clips']:
        src = clip.get('src', '')
        url = urlsplit(src)
        if not src or url.scheme or url.netloc:
            continue
        path = Path(unquote(url.path))
        if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'videos':
            raise ValueError(f'Local video paths must be inside videos/: {src}')
        origin = ROOT / path
        if not origin.is_file() or not origin.resolve().is_relative_to((ROOT / 'videos').resolve()):
            raise ValueError(f'Missing local video: {path}. Run prepare_clips.py, commit authorized video files, or use --base-url before deploying.')
        local_files.append(path)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir()
    shutil.copy2(ROOT / 'index.html', output / 'index.html')
    (output / '.nojekyll').touch()
    shutil.copytree(ROOT / 'assets', output / 'assets')
    (output / 'data').mkdir()
    shutil.copy2(ROOT / 'data/clips.json', output / 'data/clips.json')
    for path in local_files:
        (output / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, output / path)
    print(f'Built dist/ with {len(local_files)} referenced local video files.')

if __name__ == '__main__':
    build()
