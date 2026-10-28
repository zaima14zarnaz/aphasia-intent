#!/usr/bin/env python3
"""Build the clip manifest and optionally copy MP4s into the static site."""
import argparse
import json
import re
import shutil
from pathlib import Path
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path('/data/quantization/zaima/aphasiabank/neural2/control/annotations/intent/video_clips')

def natural_key(value):
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r'(\d+)', value)]

def prepare(source, manifest_path, selection=None, base_url=None, manifest_only=False):
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f'Video directory does not exist: {source}')
    if base_url and (urlsplit(base_url).scheme not in ('http', 'https') or not urlsplit(base_url).netloc):
        raise ValueError('--base-url must be an HTTP(S) directory URL.')
    if base_url and (urlsplit(base_url).query or urlsplit(base_url).fragment):
        raise ValueError('--base-url must be a directory URL without a query or fragment.')
    # Only resolve paths within the selected directory, including symlink targets.
    candidates = {path.relative_to(source).as_posix(): path for path in source.rglob('*') if path.is_file() and path.suffix.lower() == '.mp4' and path.resolve().is_relative_to(source)}
    if selection:
        selected = [line.strip() for line in selection.read_text(encoding='utf-8').splitlines() if line.strip() and not line.lstrip().startswith('#')]
        if len(set(selected)) != len(selected):
            raise ValueError('The selection file contains duplicate filenames.')
        missing = [name for name in selected if name not in candidates]
        if missing:
            raise ValueError('Selected MP4s were not found inside the source directory: ' + ', '.join(missing))
        names = selected
    else:
        names = sorted(candidates, key=natural_key)
    if not names:
        raise ValueError('No MP4 files selected; the existing manifest has not been changed.')
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    old = {clip['id']: clip for clip in previous.get('clips', [])}
    clips = []
    for i, name in enumerate(names, 1):
        prior = old.get(name, {})
        if base_url:
            src = base_url.rstrip('/') + '/' + quote(name, safe='/')
        else:
            src = 'videos/' + quote(name, safe='/')
            if not manifest_only:
                target = ROOT / 'videos' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.resolve() != candidates[name].resolve():
                    shutil.copy2(candidates[name], target)
        clips.append({
            'id': name, 'title': prior.get('title', f'Clip {i:02d}'), 'filename': name, 'src': src,
            'task': prior.get('task', 'Sandwich task'),
            'transcript': prior.get('transcript', 'Placeholder transcript — replace with the written transcript.'),
            'placeholder_context': prior.get('placeholder_context', True)
        })
    manifest = {'schema_version': 1, 'clips': clips}
    if 'examples' in previous:
        manifest['examples'] = previous['examples']
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(manifest_path)
    return clips

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE, help='MP4 folder (defaults to the requested server directory).')
    parser.add_argument('--selection', type=Path, help='UTF-8 file: one source-relative MP4 filename per line; order is preserved.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--base-url', help='Use videos hosted at this HTTP(S) directory URL instead of copying files.')
    group.add_argument('--manifest-only', action='store_true', help='Only generate the manifest; use Choose video folder at runtime.')
    args = parser.parse_args()
    try:
        clips = prepare(args.source, ROOT / 'data/clips.json', args.selection, args.base_url, args.manifest_only)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Error: {error}\n')
    print(f'Prepared {len(clips)} clips in data/clips.json.')
    if args.manifest_only:
        print('Video files were not copied. Choose the original source folder in the website.')
    elif args.base_url:
        print('Videos will load from the configured web directory.')
    else:
        print('Videos copied to videos/. Start the local HTTP server to annotate.')

if __name__ == '__main__':
    main()
